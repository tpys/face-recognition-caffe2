#include "lsoftmax_with_loss.h"


namespace caffe2 {

    REGISTER_CPU_OPERATOR(LSoftmaxWithLoss, LSoftmaxWithLossOp<float, CPUContext>);
    REGISTER_CPU_OPERATOR(LSoftmaxWithLossGradient, LSoftmaxWithLossGradientOp<float, CPUContext>);
    // Inputs: X, W, L
    // Outputs: Y, lambda, x_norm, w_norm
    OPERATOR_SCHEMA(LSoftmaxWithLoss)
                    .NumInputs(3)
                    .NumOutputs(4)
                    .TensorInferenceFunction([](const OperatorDef& def,
                                                const vector<TensorShape>& in) {

                        ArgumentHelper helper(def);
                        auto axis = helper.GetSingleArgument<int32_t>("axis", 1);
                        const auto canonical_axis = canonical_axis_index_(axis, in[0].dims().size());
                        const int K = size_from_dim_(canonical_axis, GetDimsVector(in[0])); //feature_dim
                        const int M = size_to_dim_(canonical_axis, GetDimsVector(in[0]));//batch_size
                        auto axis_w = helper.GetSingleArgument<int32_t>("axis_w", 1);
                        const auto canonical_axis_w = canonical_axis_index_(axis_w, in[1].dims().size());
                        const int N = size_to_dim_(canonical_axis_w, GetDimsVector(in[1]));//class_num

                        vector<TensorShape> out;
                        vector<TIndex> y_shape({ M, N });
                        vector<TIndex> lambda_shape({ });
                        vector<TIndex> x_norm_shape({ 1, M });
                        vector<TIndex> w_norm_shape({ 1, N });

                        out.resize(4);
                        out[0] = CreateTensorShape(y_shape, in[0].data_type());
                        out[1] = CreateTensorShape(lambda_shape, in[0].data_type());
                        out[2] = CreateTensorShape(x_norm_shape, in[0].data_type());
                        out[3] = CreateTensorShape(w_norm_shape, in[0].data_type());

                        return out;
                    });
    OPERATOR_SCHEMA(LSoftmaxWithLossGradient).NumInputs(7).NumOutputs(2);


    // Inputs: X W L dy lambda x_norm w_norm
    // Output: dx, dw
    namespace {
        class GetLSoftmaxWithLossGradient : public GradientMakerBase {
            using GradientMakerBase::GradientMakerBase;
            vector<OperatorDef> GetGradientDefs() override {
                CAFFE_ENFORCE_EQ(def_.input_size(), 3);
                return SingleGradientDef(
                        "LSoftmaxWithLossGradient",
                        "",
                        vector<string>{I(0), I(1), I(2), GO(0), O(1), O(2), O(3)},
                        vector<string>{GI(0), GI(1)}); //dX, dW
            }
        };

        REGISTER_GRADIENT(LSoftmaxWithLoss, GetLSoftmaxWithLossGradient);


        inline int pow_of_mo(const int k){ return 1 - ((k&0x01) << 1); }

        int find_k(const vector<float>& k_table, const float cos_t) {
            for(int i = k_table.size() - 1; i >=0; --i){
                if(cos_t < k_table[i]){
                    return i;
                }
            }
            return 0;
        }

        float calc_cos_mt(const vector<float>& c_table, const float cos_t, const int margin){
            const float sin2_t = 1 - cos_t * cos_t;
            float cos_t_p = std::pow(cos_t, margin);
            float sin2_t_p = 1;
            float cos_mt = cos_t_p;
            for(int p = 1; p <= margin/2; ++p){
                cos_t_p /= cos_t * cos_t;
                sin2_t_p *= sin2_t;
                cos_mt += pow_of_mo(p) * c_table[2 * p] * cos_t_p * sin2_t_p;
            }
            return cos_mt;
        }

        enum LSoftmaxTempSpaceType { kCost, kCosmt, kK, kSin2t, kFo, kCostM };
    }



    template <>
    bool LSoftmaxWithLossOp<float, CPUContext>::RunOnDevice() {
        const auto& X = Input(0);
        const auto& W = Input(1);
        const auto& L = Input(2);
        auto* Y = Output(0);
        auto* lambda = Output(1);
        auto* x_norm = Output(2);
        auto* w_norm = Output(3);


        CAFFE_ENFORCE(L.ndim() == 1, L.ndim());
        const auto canonical_axis = X.canonical_axis_index(axis_);
        const int M = X.size_to_dim(canonical_axis); //batch_size
        const int K = X.size_from_dim(canonical_axis); //feature_dim
        const auto canonical_axis_w = W.canonical_axis_index(axis_w_);
        const int N = W.size_to_dim(canonical_axis_w); //class_num

        CAFFE_ENFORCE(M == X.size() / K);
        CAFFE_ENFORCE(K == W.size() / N);
        CAFFE_ENFORCE(M == L.size());

        vector<TIndex> y_shape({ M, N });
        vector<TIndex> x_norm_shape({ 1, M });
        vector<TIndex> w_norm_shape({ 1, N });

        /**compute current lambda*/
        iter_ += 1;
        float current_lambda = base_ * powf((1.f + gamma_ * iter_), -power_);
        current_lambda = std::max(current_lambda, lambda_min_);
        lambda->Resize(vector<TIndex>());
        float* lambda_data = lambda->mutable_data<float>();
        lambda_data[0] = current_lambda;


        /**original fully connected*/
        Y->Resize(y_shape);
        float* y_mutable_data = Y->mutable_data<float>();
        const float* x_data = X.data<float>();
        const float* w_data = W.data<float>();
        math::Gemm(CblasNoTrans, CblasTrans, M, N, K, 1.f, x_data, w_data, 0.f, y_mutable_data, &context_);


        /**compute x_norm*/
        x_norm->Resize(x_norm_shape);
        float* x_norm_mutable_data = x_norm->mutable_data<float>();
        for(int i = 0; i < M; ++i){
            float res = 0;
            math::Dot(K, x_data + i * K,  x_data + i * K, &res, &context_);
            x_norm_mutable_data[i] = sqrt(res);
        }

        /**compute w_norm*/
        w_norm->Resize(w_norm_shape);
        float* w_norm_mutable_data = w_norm->mutable_data<float>();
        for(int i = 0; i < N; ++i){
            float res = 0;
            math::Dot(K, w_data + i * K,  w_data + i * K, &res, &context_);
            w_norm_mutable_data[i] = std::sqrt(res);
        }

        /**forward*/
        const int* label_data = L.data<int>();
        const float* x_norm_data = x_norm->data<float>();
        const float* w_norm_data = w_norm->data<float>();

        for(int i = 0; i < M; ++i){
            const int yi =  label_data[i];
            float fo_i_yi = y_mutable_data[i * N + yi];
            float cos_t =  fo_i_yi / (x_norm_data[i] * w_norm_data[yi]);
            int k = find_k(k_table_, cos_t);
            float cos_mt = calc_cos_mt(c_table_, cos_t, margin_);
            float f_i_yi = (pow_of_mo(k) * cos_mt - 2 * k) * (w_norm_data[yi] * x_norm_data[i]);
            y_mutable_data[i * N + yi] = (f_i_yi + current_lambda * fo_i_yi) / (1 + current_lambda);
        }

        return true;

    }

    template <>
    bool LSoftmaxWithLossGradientOp<float, CPUContext>::RunOnDevice() {
        const auto& X = Input(0);
        const auto& W = Input(1);
        const auto& L = Input(2);
        const auto& dy = Input(3);
        const auto& lambda = Input(4);
        const auto& x_norm = Input(5);
        const auto& w_norm = Input(6);
        auto* dx = Output(0);
        auto* dw = Output(1);

        // batch size
        const auto canonical_axis = X.canonical_axis_index(axis_);
        const int M = X.size_to_dim(canonical_axis);
        const int K = X.size_from_dim(canonical_axis);
        const auto canonical_axis_w = W.canonical_axis_index(axis_w_);
        const int N = W.size_to_dim(canonical_axis_w);
        CAFFE_ENFORCE(M * K == X.size());
        CAFFE_ENFORCE(K * N == W.size());

        dw->ResizeLike(W);
        dx->ResizeLike(X);

        if (X.size() == 0) {
            math::Set(dx->size(), 0.f, dx->mutable_data<float>(), &context_);
            math::Set(dw->size(), 0.f, dw->mutable_data<float>(), &context_);
            return true;
        }

        /**original fully connected*/
        float* dx_mutable_data = dx->mutable_data<float>();
        float* dw_mutable_data = dw->mutable_data<float>();
        const float* dy_data = dy.data<float>();
        const float* x_data = X.data<float>();
        const float* w_data = W.data<float>();
        math::Gemm(CblasNoTrans, CblasNoTrans, M, K, N, 1.f, dy_data, w_data, 0.f, dx_mutable_data, &context_);
        math::Gemm(CblasTrans, CblasNoTrans, N, K, M, 1.f, dy_data, x_data, 0.f, dw_mutable_data, &context_);



        /**prepare for backward*/
        const int* label_data = L.data<int>();
        const float* x_norm_data = x_norm.data<float>();
        const float* w_norm_data = w_norm.data<float>();
        const float* lambda_data = lambda.data<float>();

        vector<TIndex> workspace_shape({ 6, M });
        Tensor<CPUContext> workspace;
        workspace.Resize(workspace_shape);
        float* workspace_mutable_data = workspace.mutable_data<float>();
        const float* workspace_data = workspace.data<float>();
        for(int i = 0; i < M ; ++i){
            const int yi = label_data[i];
            float fo_i_yi = 0;
            math::Dot(K, w_data + yi * K, x_data + i * K,  &fo_i_yi, &context_);
            float cos_t = fo_i_yi / (x_norm_data[i] * w_norm_data[yi]);
            int k =  find_k(k_table_, cos_t);
            float cos_mt = calc_cos_mt(c_table_, cos_t, margin_);
            float sin2_t = 1- cos_t * cos_t;
            workspace_mutable_data[kCost * M + i] = cos_t;
            workspace_mutable_data[kCosmt * M + i] = cos_mt;
            workspace_mutable_data[kK * M + i] = static_cast<float>(k);
            workspace_mutable_data[kSin2t * M + i] = sin2_t;
            workspace_mutable_data[kFo * M + i] = fo_i_yi;
            workspace_mutable_data[kCostM * M + i] = std::pow(cos_t, margin_ - 1);
        }


        const float alpha = 1 / (1 + lambda_data[0]);

        /**backward dx*/
        for(int i = 0; i < M; ++i) {
            const int yi = label_data[i];
            const float cos_t = workspace_data[kCost * M + i];
            const float cos_mt = workspace_data[kCosmt * M + i];
            const int k = static_cast<int>(workspace_data[kK * M + i]);
            const float sin2_t = workspace_data[kSin2t * M + i];
            const float fo_i_yi = workspace_data[kFo * M + i];
            const float w_norm_yi = w_norm_data[yi];
            const float x_norm_i = x_norm_data[i];

            for(int l = 0; l < K; ++l){
                const float dcos_dx = w_data[yi * K + l] / (w_norm_yi * x_norm_i) - \
                    fo_i_yi * x_data[i * K + l] / (w_norm_yi * x_norm_i * x_norm_i * x_norm_i);
                const float dsin2_dx = -2 * cos_t * dcos_dx;
                float cos_t_p = workspace_data[kCostM * M + i];
                float sin2_t_p = 1;
                float dcosm_dx = margin_ * cos_t_p * dcos_dx;  // p = 0
                for (int p = 1; p <= margin_ / 2; ++p) {
                    cos_t_p /= cos_t * cos_t;
                    dcosm_dx += pow_of_mo(p) * c_table_[2 * p] * (p * cos_t * dsin2_dx + \
                        (margin_ - 2 * p) * sin2_t * dcos_dx) * cos_t_p * sin2_t_p;
                    sin2_t_p *= sin2_t;
                }
                const float df_dx = (pow_of_mo(k) * cos_mt - 2 * k) * w_norm_yi / x_norm_i * x_data[i * K + l] + \
                    pow_of_mo(k) * w_norm_yi * x_norm_i * dcosm_dx;
                dx_mutable_data[i * K + l] += alpha * dy_data[i * N + yi] * (df_dx - w_data[yi * K + l]);
            }
        }


        /**backward dw*/
        for(int j = 0; j < N; ++j) {
            float dw = 0;
            for(int i = 0; i < M; ++i){
                const int yi = label_data[i];
                if(yi == j){
                    const float cos_t = workspace_data[kCost * M + i];
                    const float cos_mt = workspace_data[kCosmt * M + i];
                    const int k = static_cast<int>(workspace_data[kK * M + i]);
                    const float sin2_t = workspace_data[kSin2t * M + i];
                    const float fo_i_yi = workspace_data[kFo * M + i];
                    const float x_norm_i = x_norm_data[i];
                    const float w_norm_yi = w_norm_data[yi];
                    for(int l = 0; l < K; ++l){
                        const float dcos_dw = x_data[i * K + l] / (w_norm_yi * x_norm_i) - \
                            fo_i_yi * w_data[yi * K + l] / (x_norm_i * w_norm_yi * w_norm_yi * w_norm_yi);
                        const float dsin2_dw = -2 * cos_t * dcos_dw;
                        float cos_t_p = workspace_data[kCostM * M + i];
                        float sin2_t_p = 1;
                        float dcosm_dw = margin_ * cos_t_p * dcos_dw;  // p = 0
                        for (int p = 1; p <= margin_ / 2; ++p) {
                            cos_t_p /= cos_t * cos_t;
                            dcosm_dw += pow_of_mo(p) * c_table_[2 * p] * (p * cos_t * dsin2_dw + \
                                (margin_ - 2 * p) * sin2_t * dcos_dw) * cos_t_p * sin2_t_p;
                            sin2_t_p *= sin2_t;
                        }
                        const float df_dw_j = (pow_of_mo(k) * cos_mt - 2 * k) * \
                            x_norm_i / w_norm_yi * w_data[yi * K +l] + \
                            pow_of_mo(k) * w_norm_yi * x_norm_i * dcosm_dw;
                        dw_mutable_data[j * K + l] += alpha * dy_data[i * N + yi] * (df_dw_j - x_data[i * K + l]);
                    }
                }
            }
        }

        return true;
    }



}  // namespace caffe2

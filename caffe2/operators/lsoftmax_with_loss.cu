#include "lsoftmax_with_loss.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {
    namespace {

        enum LSoftmaxTempSpaceType { kCost, kCosmt, kK, kSin2t, kFo, kCostM };

        __device__ __forceinline__  int LSPowOfMO(const int k) {
            return 1 - ((k&0x01) << 1);
        }


        template<typename DType>
        __global__ void LSCalcNorm(int nthreads, const int K, const DType* data, DType* data_norm) {
            CUDA_1D_KERNEL_LOOP(index, nthreads) {
                DType sum_sqaure = 0.;
                for (int j = 0; j < K; j++) {
                    sum_sqaure += data[index * K + j] * data[index * K + j];
                }
                data_norm[index] = sqrt(sum_sqaure);
            }
        }


        template<typename DType>
        __device__ int LSFindK(const DType *k_table, const int N, const DType cos_t) {
            for(int i = N - 1; i >=0; --i){
                if(cos_t < k_table[i]){
                    return i;
                }
            }
            return 0;
        }


        template<typename DType>
        __device__ DType LSCalcCosmt(const DType *c_table, const DType cos_t, const int margin) {
            const DType sin2_t = 1 - cos_t * cos_t;
            DType cos_t_p = pow(cos_t, margin);
            DType sin2_t_p = 1;
            DType cos_mt = cos_t_p;  // p = 0
            for (int p = 1; p <= margin / 2; ++p) {
                // don't replace `cos_t*cos_t` with `1-sin2_t`, this can cause numeric issue if cos_t --> 0
                cos_t_p /= cos_t * cos_t;
                sin2_t_p *= sin2_t;
                cos_mt += LSPowOfMO(p) * c_table[2 * p] * cos_t_p * sin2_t_p;
            }
            return cos_mt;
        }


        template<typename DType>
        __global__ void LSoftmaxForwardKernel(const int nthreads,
                                              const int N,
                                              const int K,
                                              const int NTable,
                                              const int margin,
                                              const float lambda,
                                              const float* k_table,
                                              const float* c_table,
                                              const int* label,
                                              const DType* x,
                                              const DType* w,
                                              const DType* x_norm,
                                              const DType* w_norm,
                                              DType* out) {
            CUDA_1D_KERNEL_LOOP(index, nthreads) {
                const int yi = label[index];
                const DType fo_i_yi = out[index * N + yi];
                const DType cos_t = fo_i_yi / (x_norm[index] * w_norm[yi]);
                const int k = LSFindK(k_table, NTable, cos_t);
                const DType cos_mt = LSCalcCosmt(c_table, cos_t, margin);
                const DType f_i_yi = (LSPowOfMO(k) * cos_mt - 2 * k) * (w_norm[yi] * x_norm[index]);
                out[index * N + yi] = (f_i_yi + lambda * fo_i_yi) / (1 + lambda);
            }
        }


        template<typename DType>
        __global__ void LSoftmaxBackwardRequired(const int nthreads,
                                                 const int N,
                                                 const int K,
                                                 const int NTable,
                                                 const int margin,
                                                 const float* k_table,
                                                 const float* c_table,
                                                 const int* label,
                                                 const DType* x,
                                                 const DType* w,
                                                 const DType* x_norm,
                                                 const DType* w_norm,
                                                 DType* workspace) {
            CUDA_1D_KERNEL_LOOP(index, nthreads) {
                const int yi = label[index];

                DType fo_i_yi = 0;
                for (int p = 0; p < K; ++p) {
                    fo_i_yi += w[yi * K + p] * x[index * K + p];
                }

                const DType cos_t = fo_i_yi / (x_norm[index] * w_norm[yi]);
                const int k = LSFindK(k_table, NTable, cos_t);
                const DType cos_mt = LSCalcCosmt(c_table, cos_t, margin);
                const DType sin2_t = 1 - cos_t * cos_t;

                workspace[kCost * nthreads + index] = cos_t;
                workspace[kCosmt * nthreads + index] = cos_mt;
                workspace[kK * nthreads + index] = static_cast<DType>(k);
                workspace[kSin2t * nthreads + index] = sin2_t;
                workspace[kFo * nthreads + index] = fo_i_yi;
                workspace[kCostM * nthreads + index] = pow(cos_t, margin - 1);
            }
        }


        template<typename DType>
        __global__ void LSoftmaxBackwardXKernel(const int M,
                                                const int N,
                                                const int K,
                                                const int NTable,
                                                const int margin,
                                                const float* lambda,
                                                const float* c_table,
                                                const int* label,
                                                const DType* x,
                                                const DType* w,
                                                const DType* x_norm,
                                                const DType* w_norm,
                                                const DType* o_grad,
                                                const DType* workspace,
                                                DType* x_grad) {
            const int nthreads = M * K;
            CUDA_1D_KERNEL_LOOP(index, nthreads) {
                const int i = index / K;
                const int l = index % K;
                const int yi = label[i];
                const DType cos_t = workspace[kCost * M + i];
                const DType cos_mt = workspace[kCosmt * M + i];
                const int k = static_cast<int>(workspace[kK * M + i]);
                const DType sin2_t = workspace[kSin2t * M + i];
                const DType fo_i_yi = workspace[kFo * M + i];
                const DType w_norm_yi = w_norm[yi];
                const DType x_norm_i = x_norm[i];

                const DType dcos_dx = w[yi * K + l] / (w_norm_yi * x_norm_i) - \
                        fo_i_yi * x[i * K + l] / (w_norm_yi * x_norm_i * x_norm_i * x_norm_i);
                const DType dsin2_dx = -2 * cos_t * dcos_dx;
                DType cos_t_p = workspace[kCostM * M + i];
                DType sin2_t_p = 1;
                DType dcosm_dx = margin * cos_t_p * dcos_dx;  // p = 0
                for (int p = 1; p <= margin / 2; ++p) {
                    cos_t_p /= cos_t * cos_t;
                    dcosm_dx += LSPowOfMO(p) * c_table[2 * p] * (p * cos_t * dsin2_dx + \
                            (margin - 2 * p) * sin2_t * dcos_dx) * cos_t_p * sin2_t_p;
                    sin2_t_p *= sin2_t;
                }
                const DType df_dx = (LSPowOfMO(k) * cos_mt - 2 * k) * w_norm_yi / x_norm_i * x[i * K + l] + \
                        LSPowOfMO(k) * w_norm_yi * x_norm_i * dcosm_dx;
                const DType alpha = 1 / (1 + lambda[0]);
                x_grad[i * K + l] += alpha * o_grad[i * N + yi] * (df_dx - w[yi * K + l]);
            }
        }


        template<typename DType>
        __global__ void LSoftmaxBackwardWKernel(const int M,
                                                const int N,
                                                const int K,
                                                const int NTable,
                                                const int margin,
                                                const float* lambda,
                                                const float* c_table,
                                                const int* label,
                                                const DType* x,
                                                const DType* w,
                                                const DType* x_norm,
                                                const DType* w_norm,
                                                const DType* o_grad,
                                                const DType* workspace,
                                                DType* w_grad) {
            const int nthreads = N * K;
            CUDA_1D_KERNEL_LOOP(index, nthreads) {
                const int j = index / K;
                const int l = index % K;
                DType dw = 0;
                for (int i = 0; i < M; ++i) {
                    const int yi = label[i];
                    if (yi == j) {
                        const DType cos_t = workspace[kCost * M + i];
                        const DType cos_mt = workspace[kCosmt * M + i];
                        const int k = static_cast<int>(workspace[kK * M + i]);
                        const DType sin2_t = workspace[kSin2t * M + i];
                        const DType fo_i_yi = workspace[kFo * M + i];
                        const DType x_norm_i = x_norm[i];
                        const DType w_norm_yi = w_norm[yi];

                        const DType dcos_dw = x[i * K + l] / (w_norm_yi * x_norm_i) - \
                                  fo_i_yi * w[yi * K + l] / (x_norm_i * w_norm_yi * w_norm_yi * w_norm_yi);
                        const DType dsin2_dw = -2 * cos_t * dcos_dw;
                        DType cos_t_p = workspace[kCostM * M + i];
                        DType sin2_t_p = 1;
                        DType dcosm_dw = margin * cos_t_p * dcos_dw;  // p = 0
                        for (int p = 1; p <= margin / 2; ++p) {
                            cos_t_p /= cos_t * cos_t;
                            dcosm_dw += LSPowOfMO(p) * c_table[2 * p] * (p * cos_t * dsin2_dw + \
                            (margin - 2 * p) * sin2_t * dcos_dw) * cos_t_p * sin2_t_p;
                            sin2_t_p *= sin2_t;
                        }
                        const DType df_dw_j = (LSPowOfMO(k) * cos_mt - 2 * k) * x_norm_i / w_norm_yi * w[yi * K +l] + \
                                   LSPowOfMO(k) * w_norm_yi * x_norm_i * dcosm_dw;
                        dw += o_grad[i * N + yi] * (df_dw_j - x[i * K + l]);
                    }
                }
                const DType alpha = 1 / (1 + lambda[0]);
                w_grad[j * K + l] += alpha * dw;
            }
        }



    }  // namespace



    template <>
    bool LSoftmaxWithLossOp<float, CUDAContext>::RunOnDevice() {
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
        current_lambda = max(current_lambda, lambda_min_);
        lambda->Resize(vector<TIndex>());
        float* lambda_data = lambda->mutable_data<float>();
        // math::Set(1, current_lambda, lambda_data, &context_);
        context_.Copy<float, CPUContext, CUDAContext>(1, &current_lambda, lambda_data);


        /**original fully connected*/
        Y->Resize(y_shape);
        float* y_mutable_data = Y->mutable_data<float>();
        const float* x_data = X.data<float>();
        const float* w_data = W.data<float>();
        math::Gemm(CblasNoTrans, CblasTrans, M, N, K, 1.f, x_data, w_data, 0.f, y_mutable_data, &context_);

        /**compute x_norm*/
        x_norm->Resize(x_norm_shape);
        float* x_norm_mutable_data = x_norm->mutable_data<float>();
        LSCalcNorm<<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS>>>(M, K, x_data, x_norm_mutable_data);

        /**compute w_norm*/
        w_norm->Resize(w_norm_shape);
        float* w_norm_mutable_data = w_norm->mutable_data<float>();
        LSCalcNorm<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, K, w_data, w_norm_mutable_data);

        /**forward*/
        const int* label_data = L.data<int>();
        const float* x_norm_data = x_norm->data<float>();
        const float* w_norm_data = w_norm->data<float>();
        const int NTable = k_table_.size();
        vector<TIndex> table_shape({ 1, NTable });
        Tensor<CUDAContext> k_table;
        Tensor<CUDAContext> c_table;
        k_table.Resize(table_shape);
        c_table.Resize(table_shape);

        float* k_multable_data = k_table.mutable_data<float>();
        float* c_multable_data = c_table.mutable_data<float>();
        context_.Copy<float, CPUContext, CUDAContext>(NTable, k_table_.data(), k_multable_data);
        context_.Copy<float, CPUContext, CUDAContext>(NTable, c_table_.data(), c_multable_data);
        const float* k_data = k_table.data<float>();
        const float* c_data = c_table.data<float>();
        LSoftmaxForwardKernel<<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS>>>(M, N, K, NTable, margin_, current_lambda,
                k_data,
                c_data,
                label_data,
                x_data,
                w_data,
                x_norm_data,
                w_norm_data,
                y_mutable_data);
        return true;
    }


    template <>
    bool LSoftmaxWithLossGradientOp<float, CUDAContext>::RunOnDevice() {
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
        const int NTable = k_table_.size();
        vector<TIndex> table_shape({1, NTable});
        Tensor<CUDAContext> k_table;
        Tensor<CUDAContext> c_table;
        k_table.Resize(table_shape);
        c_table.Resize(table_shape);

        float* k_multable_data = k_table.mutable_data<float>();
        float* c_multable_data = c_table.mutable_data<float>();

        context_.Copy<float, CPUContext, CUDAContext>(NTable, k_table_.data(), k_multable_data);
        context_.Copy<float, CPUContext, CUDAContext>(NTable, c_table_.data(), c_multable_data);

        const float* k_data = k_table.data<float>();
        const float* c_data = c_table.data<float>();
        const int* label_data = L.data<int>();
        const float* x_norm_data = x_norm.data<float>();
        const float* w_norm_data = w_norm.data<float>();
        const float* lambda_data = lambda.data<float>();

        vector<TIndex> workspace_shape({ 6, M });
        Tensor<CUDAContext> workspace;
        workspace.Resize(workspace_shape);

        float* workspace_mutable_data = workspace.mutable_data<float>();
        const float* workspace_data = workspace.data<float>();
        LSoftmaxBackwardRequired<<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS>>>(M,
                N,
                K,
                NTable,
                margin_,
                k_data,
                c_data,
                label_data,
                x_data,
                w_data,
                x_norm_data,
                w_norm_data,
                workspace_mutable_data);

        /**backward dx*/
        LSoftmaxBackwardXKernel<<<CAFFE_GET_BLOCKS((M * K)), CAFFE_CUDA_NUM_THREADS>>>(M,
                N,
                K,
                NTable,
                margin_,
                lambda_data,
                c_data,
                label_data,
                x_data,
                w_data,
                x_norm_data,
                w_norm_data,
                dy_data,
                workspace_data,
                dx_mutable_data);

        /**backward dw*/
        LSoftmaxBackwardWKernel<<<CAFFE_GET_BLOCKS((N * K)), CAFFE_CUDA_NUM_THREADS>>>(M,
                N,
                K,
                NTable,
                margin_,
                lambda_data,
                c_data,
                label_data,
                x_data,
                w_data,
                x_norm_data,
                w_norm_data,
                dy_data,
                workspace_data,
                dw_mutable_data);

        return true;
    }


    REGISTER_CUDA_OPERATOR(LSoftmaxWithLoss, LSoftmaxWithLossOp<float, CUDAContext>);
    REGISTER_CUDA_OPERATOR(LSoftmaxWithLossGradient, LSoftmaxWithLossGradientOp<float, CUDAContext>);
}  // namespace caffe2

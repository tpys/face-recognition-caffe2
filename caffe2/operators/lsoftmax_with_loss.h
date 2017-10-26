#ifndef LSOFTMAX_WITH_LOSS_OP_H
#define LSOFTMAX_WITH_LOSS_OP_H

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {



    template <typename T,  class Context>
    class LSoftmaxWithLossOp: public Operator<Context>{
    public:
        USE_OPERATOR_CONTEXT_FUNCTIONS;
        LSoftmaxWithLossOp(const OperatorDef& operator_def, Workspace* ws)
                : Operator<Context>(operator_def, ws),
                  OP_SINGLE_ARG(int, "margin", margin_, 0),
                  OP_SINGLE_ARG(int, "axis", axis_, 1),
                  OP_SINGLE_ARG(int, "axis_w", axis_w_, 1),
                  OP_SINGLE_ARG(int, "iter", iter_, 0),
                  OP_SINGLE_ARG(float, "base", base_, 0),
                  OP_SINGLE_ARG(float, "gamma", gamma_, 0.12),
                  OP_SINGLE_ARG(float, "power", power_, 1),
                  OP_SINGLE_ARG(float, "lambda_min", lambda_min_, 0) {
            k_table_.clear();
            c_table_.clear();
            k_table_.push_back(1);
            c_table_.push_back(1);
            const double pi = std::atan(1) * 4;
            double factor = 1;
            for (int i = 1; i <= margin_; ++i) {
                factor = factor * (margin_ - i + 1) / i;
                k_table_.push_back(std::cos(i * pi / margin_));
                c_table_.push_back(factor);
            }
        }

        ~LSoftmaxWithLossOp() {}
        bool RunOnDevice() override;

    protected:
        int margin_;
        int iter_;
        int axis_;
        int axis_w_;
        float base_;
        float gamma_;
        float power_;
        float lambda_min_;

        vector<float> k_table_;
        vector<float> c_table_;
    };




    template <typename T, class Context>
    class LSoftmaxWithLossGradientOp: public Operator<Context>{
    public:
        LSoftmaxWithLossGradientOp(const OperatorDef& operator_def, Workspace* ws):
                Operator<Context>(operator_def, ws),
                OP_SINGLE_ARG(int, "margin", margin_, 0),
                OP_SINGLE_ARG(int, "axis", axis_, 1),
                OP_SINGLE_ARG(int, "axis_w", axis_w_, 1){
            k_table_.clear();
            c_table_.clear();
            k_table_.push_back(1);
            c_table_.push_back(1);
            const double pi = std::atan(1) * 4;
            double factor = 1;
            for (int i = 1; i <= margin_; ++i) {
                factor = factor * (margin_ - i + 1) / i;
                k_table_.push_back(std::cos(i * pi / margin_));
                c_table_.push_back(factor);
            }
        }


        USE_OPERATOR_CONTEXT_FUNCTIONS;
        bool RunOnDevice() override;


    protected:
        int margin_;
        int axis_;
        int axis_w_;

        vector<float> k_table_;
        vector<float> c_table_;
    };



}


#endif //LSOFTMAX_WITH_LOSS_OP_H

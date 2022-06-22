#pragma once
#include "fast_conv1d.h"


namespace FastConv
{

    template <typename T, int inSizet, int outSizet, int kernelSize, int dilation>
    class Conv1DStaticGated
    {
        using vecType = Eigen::Vector<T, outSizet>;
    public:
        static constexpr auto inSize = inSizet;
        static constexpr auto outSize = outSizet;
        static constexpr auto innerSize = inSizet * 2;
        static constexpr auto memorySize = (kernelSize - 1) * dilation + 1;

        Conv1DStaticGated()
        {
            reset();
        }

        void reset() 
        {
            innerOuts.resize(innerSize);
        }

        Conv1DT<T, inSize, innerSize, kernelSize, dilation> inConv;
        Conv1DT<T, inSize, outSize, 1, 1> outConv;

        std::vector<T> innerOuts;
        vecType tanhOuts;
        vecType sigmOuts;
        
        vecType outs;

        inline void forward(const T* in, T* out, T* skip, int skipIdx)
        {
            inConv.forward(in, innerOuts.data());
            tanhOuts = Eigen::Map<Eigen::Vector<T, inSize>>(innerOuts.data(), outSize);
            sigmOuts = Eigen::Map<Eigen::Vector<T, inSize>>(innerOuts.data() + outSize, outSize);
            
            tanhOuts = tanhOuts.array().tanh();
            sigmOuts = 1.0f / (1.0f + (-sigmOuts.array().exp()));
            
            outs = tanhOuts.cwiseProduct(sigmOuts);
            
            std::copy(outs.data(), outs.data() + outSize, skip + skipIdx);

            outConv.forward(outs.data(), outs.data());
            outs += Eigen::Map<const Eigen::Vector<T, outSize>>(in, outSize);            
           
            std::copy(outs.data(), outs.data() + outSize, out);
        }
    };


    template <typename T>
    class Conv1DGated
    {
        using vecType = Eigen::Vector<T, Eigen::Dynamic>;
    public:

        Conv1DGated(size_t inSize, size_t outSize, size_t kernelSize, int dilation) :
            inSize(inSize),
            outSize(outSize),
            kernelSize(kernelSize),
            innerSize(outSize * 2),
            dilation(dilation)
        {
            reset();
        }

        void reset()
        {
            innerOuts.resize(innerSize);
            inConv.reset(inSize, innerSize, kernelSize, dilation);
            outConv.reset(inSize, outSize, 1, 1);
            tanhOuts.resize(outSize);
            sigmOuts.resize(outSize);
            outs.resize(outSize);
        }

        Conv1D<T> inConv = Conv1D<T>(1, 1, 1, 1);
        Conv1D<T> outConv = Conv1D<T>(1, 1, 1, 1);

        size_t inSize;
        size_t outSize;
        size_t kernelSize;
        size_t innerSize;
        int dilation;

        std::vector<T> innerOuts;
        vecType tanhOuts;
        vecType sigmOuts;
        vecType outs;

        inline void forward(const T* in, T* out, T* skip, int skipIdx)
        {
            inConv.forward(in, innerOuts.data());
            tanhOuts = Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>>(innerOuts.data(), outSize);
            sigmOuts = Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>>(innerOuts.data() + outSize, outSize);

            tanhOuts = tanhOuts.array().tanh();
            sigmOuts = 1.0f / (1.0f + (-sigmOuts.array().exp()));

            outs = tanhOuts.cwiseProduct(sigmOuts);
            std::copy(outs.data(), outs.data() + outSize, skip + skipIdx);

            outConv.forward(outs.data(), outs.data());
            outs += Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>>(in, outSize);

            std::copy(outs.data(), outs.data() + outSize, out);
        }
    };

} // FastConv

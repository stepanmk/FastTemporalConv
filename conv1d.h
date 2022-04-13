#pragma once
#include <Eigen/Dense>
#include <vector>

namespace FastConv
{

    template <typename T, int inSizet, int outSizet, int kernelSize, int dilation>
    class Conv1DStatic
    {

        static constexpr auto memorySize = (kernelSize - 1) * dilation + 1;
        using vecType = Eigen::Vector<T, outSizet>;
        using filterType = Eigen::Matrix<T, inSizet, kernelSize>;
        using memoryType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
        using memoryPtrsType = Eigen::Vector<int, kernelSize>;

    public:
        static constexpr auto inSize = inSizet;
        static constexpr auto outSize = outSizet;
        

        Conv1DStatic()
        {
            for (int i = 0; i < outSize; ++i)
                weights[i] = filterType::Random();
            inCols.setZero();
            bias = vecType::Random();
            reset();
        }

        void reset() {
            memoryPtr = 0;
            memory.resize(inSizet, memorySize);
            memory.setZero();
        }

        void setWeights(const std::vector<std::vector<std::vector<T>>>& w)
        {
            for (int i = 0; i < outSize; ++i)
                for (int k = 0; k < inSize; ++k)
                    for (int j = 0; j < kernelSize; ++j)
                        weights[i](k, j) = w[i][k][j];
        }

        void setBias(const std::vector<T>& b)
        {
            for (int i = 0; i < outSize; ++i)
                bias(i) = b[i];
        }

        inline void forward(const T* in, T* out)
        {
            memory.col(memoryPtr) = Eigen::Map<const Eigen::Vector<T, inSize>>(in, inSize);
            setColPointers(memoryColPtrs);

            for (int k = 0; k < kernelSize; ++k)
            {
                inCols.col(k) = memory.col(memoryColPtrs(k));
            }

            for (int i = 0; i < outSize; ++i)
                out[i] = inCols.cwiseProduct(weights[i]).sum() + bias(i);
            
            memoryPtr = mod(memoryPtr + 1, memorySize);
        }

    private:

        memoryType memory;
        int memoryPtr = 0;

        filterType weights[outSize];
        vecType bias;

        memoryPtrsType memoryColPtrs;
        filterType inCols;

        int mod(int a, int b)
        {
            int r = a % b;
            return r < 0 ? r + b : r;
        }

        inline void setColPointers(memoryPtrsType& ptr)
        {
            for (int i = 0; i < kernelSize; ++i)
                ptr(i) = mod((memoryPtr - i * dilation), memorySize);
        }
    };

    template <typename T>
    class Conv1D
    {
        using vecType = Eigen::Vector<T, Eigen::Dynamic>;
        using filterType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
        using memoryType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
        using memoryPtrsType = Eigen::VectorXi;

    public:

        size_t inSize;
        size_t outSize;
        size_t kernelSize;
        size_t memorySize;
        int dilation;

        Conv1D(size_t inSize, size_t outSize, size_t kernelSize, int dilation) :
            inSize(inSize),
            outSize(outSize),
            kernelSize(kernelSize),
            memorySize((kernelSize - 1)* dilation + 1),
            dilation(dilation)
        {
            weights.resize(outSize);
            for (int i = 0; i < outSize; ++i)
                weights[i] = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Random(inSize, kernelSize);
            bias = Eigen::Vector<T, Eigen::Dynamic>::Random(outSize);

            memory = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Random(inSize, memorySize);
            memoryColPtrs = Eigen::VectorXi::Zero(kernelSize);
            inCols = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(inSize, kernelSize);
            outs.resize(outSize);
            outs.setZero();
        }

        /** Resets the layer state. */
        void reset(size_t newInSize, size_t newOutSize, size_t newKernelSize, int newDilation)
        {
            inSize = newInSize;
            outSize = newOutSize;
            kernelSize = newKernelSize;
            dilation = newDilation;
            memorySize = (kernelSize - 1) * dilation + 1;

            weights.resize(outSize);
            for (int i = 0; i < outSize; ++i)
                weights[i] = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Random(inSize, kernelSize);

            bias = Eigen::Vector<T, Eigen::Dynamic>::Random(outSize);

            memory = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(inSize, memorySize);
            memoryColPtrs = Eigen::VectorXi::Zero(kernelSize);
            
            inCols = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(inSize, kernelSize);
            
            outs.resize(outSize);
            outs.setZero();
            memoryPtr = 0;
        }

        void setWeights(const std::vector<std::vector<std::vector<T>>>& w)
        {
            for (int i = 0; i < outSize; ++i)
                for (int k = 0; k < inSize; ++k)
                    for (int j = 0; j < kernelSize; ++j)
                        weights[i](k, j) = w[i][k][j];
        }

        void setBias(const std::vector<T>& b)
        {
            for (int i = 0; i < outSize; ++i)
                bias(i) = b[i];
        }

        inline void forward(const T* in, T* out)
        {
            memory.col(memoryPtr) = Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>>(in, inSize);
            setColPointers(memoryColPtrs);

            for (int k = 0; k < kernelSize; ++k)
            {
                inCols.col(k) = memory.col(memoryColPtrs(k));
            }

            for (int i = 0; i < outSize; ++i)
                out[i] = inCols.cwiseProduct(weights[i]).sum() + bias(i);

            memoryPtr = mod(memoryPtr + 1, memorySize);
        }

    private:

        memoryType memory;
        int memoryPtr = 0;

        std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> weights;
        vecType bias;

        memoryPtrsType memoryColPtrs;
        filterType inCols;

        int mod(int a, int b)
        {
            int r = a % b;
            return r < 0 ? r + b : r;
        }

        inline void setColPointers(memoryPtrsType& ptr)
        {
            for (int i = 0; i < kernelSize; ++i)
                ptr(i) = mod((memoryPtr - i * dilation), memorySize);
        }
    };

} // FastConv

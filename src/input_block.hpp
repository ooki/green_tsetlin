#ifndef __INPUT_BLOCK_HPP_
#define __INPUT_BLOCK_HPP_

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdint.h>

#include <gt_common.hpp>

namespace green_tsetlin
{


    class InputBlock
    {
        public:
            virtual ~InputBlock() {}

            virtual void prepare_example(int index) {}
            virtual const uint32_t* pull_current_label() const
            {
                return nullptr;
            }
            virtual int get_number_of_examples() const 
            {
                return -42;
            }

            virtual bool is_multi_label() const 
            {
                return get_num_labels_per_example() > 1;
            }

            virtual int get_num_labels_per_example() const 
            {
                return 0;
            }
    };


    template <typename _ExampleType>
    class DenseInputBlock : public InputBlock
    {
        public:
            typedef _ExampleType example_type;

            DenseInputBlock(int num_literals)
            {   
                const int align_to = 32;
                m_num_labels_per_example = 0;

                m_num_literals = num_literals;
                m_align_example_to = align_to;                
                                                                       
                m_num_examples = 0;
                int one_example_mem = (((num_literals * sizeof(_ExampleType)) / align_to) + 1) * align_to;

                m_labels = nullptr;
                m_data = nullptr;

                // allocate aligned mem for 1 data point
                m_current_example = reinterpret_cast<_ExampleType*>(std::aligned_alloc(align_to, one_example_mem));

                for(int i = 0; i  < one_example_mem; ++i)
                    m_current_example[i] = 0;
            }

            virtual ~DenseInputBlock()
            {
                if(m_current_example != nullptr)
                {
                    free(m_current_example);
                    m_current_example = nullptr;
                }
            }

            virtual int get_number_of_examples() const 
            {
                return m_num_examples;
            }

            virtual void prepare_example(int index)
            {
                if(m_labels != nullptr)                                      
                    m_current_label = &m_labels[index*m_num_labels_per_example];                
                    
                memcpy(m_current_example, &m_data[index * m_num_literals], m_num_literals);
            }

            uint8_t* pull_current_example() const
            {
                return m_current_example;
            }

            virtual const uint32_t* pull_current_label() const
            {
                return m_current_label;
            }

            virtual int get_num_labels_per_example() const 
            {
                return m_num_labels_per_example;
            }    

            void set_data(pybind11::array_t<uint8_t> examples, pybind11::array_t<uint32_t> labels)
            {
                //data 
                pybind11::buffer_info buffer_info = examples.request();                            
                std::vector<ssize_t> shape = buffer_info.shape;
                m_num_examples = shape[0];
  
                if(shape[1] != m_num_literals)
                    throw std::runtime_error("Number of literals does not match the data provided in set_data().");

                m_data = static_cast<example_type*>(buffer_info.ptr);

                // labels
                pybind11::buffer_info buffer_info2 = labels.request();                            
                std::vector<ssize_t> shape2 = buffer_info2.shape;      
                if(shape2.size() == 0 || shape2[0] == 0)
                {
                    m_labels = nullptr;
                }           
                else
                {
                    if(shape2[0] != m_num_examples)
                        throw std::runtime_error("Number of examples in labeles does not match number of examples provided in set_data().");
                    
                    //std::cout << "shape2.size(): " << shape2.size() << "  [0] = " << shape2[0] << std::endl;
                    if(shape2.size() == 1)                    
                        m_num_labels_per_example = 1;                    
                    else                    
                        m_num_labels_per_example = shape2[1];
                    
                    m_labels = static_cast<uint32_t*>(buffer_info2.ptr);
                }
            }

        protected:
            int m_num_literals;
            int m_num_examples;
            int m_align_example_to;            
            int m_num_labels_per_example;

            example_type*               m_data;            
            uint32_t*                   m_labels;

            example_type*               m_current_example = nullptr;
            uint32_t*                   m_current_label;
    };

    
    template <typename _ExampleType>
    class SparseInputDenseOutputBlock : public DenseInputBlock<_ExampleType>
    {
        public:
            SparseInputDenseOutputBlock(int num_literals) : DenseInputBlock<_ExampleType>(num_literals) {}        
            virtual ~SparseInputDenseOutputBlock() {}

            
            virtual void prepare_example(int index)
            {
                if(this->m_labels != nullptr)
                    this->m_current_label = &this->m_labels[index*this->m_num_labels_per_example];

                memset(this->m_current_example, 0, this->m_num_literals);

                const int32_t row_end = m_indptr[index+1];
                for(int32_t row_iter = m_indptr[index]; row_iter < row_end; row_iter++)                                    
                {
                    this->m_current_example[m_indices[row_iter]] = 1;
                }   
            }
            
            void set_data_sparse(pybind11::array indices, pybind11::array indptr, pybind11::array labels)
            {
                // 
                pybind11::buffer_info indices_info = indices.request();                            
                m_indices = static_cast<int32_t*>(indices_info.ptr);

                pybind11::buffer_info indptr_info = indptr.request();                            
                std::vector<ssize_t> shape_indptr = indptr_info.shape;

                this->m_num_examples = shape_indptr[0] - 1;
                m_indptr = static_cast<int32_t*>(indptr_info.ptr);

                // labels
                pybind11::buffer_info buffer_info2 = labels.request();                            
                std::vector<ssize_t> shape2 = buffer_info2.shape;      
                
                if(shape2[0] != this->m_num_examples)
                    throw std::runtime_error("Number of examples in labeles does not match number of examples provided in set_data().");
                
                //std::cout << "shape2.size(): " << shape2.size() << "  [0] = " << shape2[0] << std::endl;
                if(shape2.size() == 1)                    
                    this->m_num_labels_per_example = 1;                    
                else                    
                    this->m_num_labels_per_example = shape2[1];
                
                this->m_labels = static_cast<uint32_t*>(buffer_info2.ptr);            
            }
            void set_data(pybind11::array_t<uint8_t> examples, pybind11::array_t<uint32_t> labels)
            {
                throw std::runtime_error("Cannot use dense input for SparseInputDenseOutputBlock.");
            }
            
            
        protected:
            int32_t*                    m_indices;
            int32_t*                    m_indptr;
        };
    


    template <typename _ExampleType>
    class SparseInputBlock : public InputBlock
    {
        public:
            typedef _ExampleType example_type;

            SparseInputBlock(int num_literals)
            {   
                const int align_to = 32;

                m_num_literals = num_literals;
                m_align_example_to = align_to;                
                                             
                m_num_examples = 0;
                m_num_labels_per_example = 0;
                m_labels = nullptr;
                m_indices = nullptr;
                m_indptr = nullptr;
                
                m_current_example.reserve(num_literals); // TODO: do we need to have memory alloc for a full example here if it is sparse?
            }

            virtual ~SparseInputBlock()
            {            
            }

            virtual int get_number_of_examples() const 
            {
                return m_num_examples;
            }

            virtual void prepare_example(int index)
            {
                if(m_labels != nullptr)
                    m_current_label = &m_labels[index*m_num_labels_per_example];

                m_current_example.clear();

                const int32_t row_end = m_indptr[index+1];
                for(int32_t row_iter = m_indptr[index]; row_iter < row_end; row_iter++)
                {
                    // change here if you change SparseLiteral def (top of sparse_tsetlin_state.hpp)
                    
                    m_current_example.push_back(m_indices[row_iter]); // vector 
                    // m_current_example.insert(m_indices[row_iter]); // unordered_map
                }
                    
            }

            _ExampleType* pull_current_example()
            {
                return &m_current_example;
            }

            virtual const uint32_t* pull_current_label() const
            {
                return m_current_label;
            }

            void set_data(pybind11::array indices, pybind11::array indptr, pybind11::array labels)
            {
                // 
                pybind11::buffer_info indices_info = indices.request();                            
                m_indices = static_cast<int32_t*>(indices_info.ptr);

                pybind11::buffer_info indptr_info = indptr.request();                            
                std::vector<ssize_t> shape_indptr = indptr_info.shape;

                m_num_examples = shape_indptr[0] - 1;
                m_indptr = static_cast<int32_t*>(indptr_info.ptr);

                // labels
                pybind11::buffer_info buffer_info2 = labels.request();                            
                std::vector<ssize_t> shape2 = buffer_info2.shape;      
                
                if(shape2[0] != m_num_examples)
                    throw std::runtime_error("Number of examples in labeles does not match number of examples provided in set_data().");
                
                //std::cout << "shape2.size(): " << shape2.size() << "  [0] = " << shape2[0] << std::endl;
                if(shape2.size() == 1)                    
                    m_num_labels_per_example = 1;                    
                else                    
                    m_num_labels_per_example = shape2[1];
                
                m_labels = static_cast<uint32_t*>(buffer_info2.ptr);
            
            }
            
        protected:
            int m_num_literals;
            int m_num_examples;
            int m_align_example_to;            
            int m_num_labels_per_example;

            int32_t*                    m_indices;
            int32_t*                    m_indptr;
            uint32_t*                   m_labels;

            example_type               m_current_example;
            uint32_t*                  m_current_label;
    };


    inline uint8_t* cumlative_encode(uint8_t* out, uint32_t number, uint32_t size)
    {
        std::memset(out, 0, size);
        for(uint32_t i = 0; i < number; ++i)      
            out[i] = 1;        
        
        return out + size;
    }

    pybind11::array_t<uint8_t> tsetlin_im2col(pybind11::array_t<uint8_t> numpy_examples, int patch_width, int patch_height)
    {
        /**
         * A function for converting input image patches to a column matrix for the (Convolutional) Tsetlin Machine.
         *
         * @param numpy_examples input examples in a NumPy array
         * @param patch_width width of the image patch
         * @param patch_height height of the image patch         
         *
         * @return a NumPy array representing the column matrix
         *
         * @throws None
         */
        
        //data 
        pybind11::buffer_info buffer_info = numpy_examples.request();                            
        std::vector<ssize_t> shape = buffer_info.shape;
        
        const int n_examples = shape[0];
        const int width = shape[1];
        const int height = shape[2];
        const int channels = shape[3];
        uint8_t* src = static_cast<uint8_t*>(buffer_info.ptr);

        // no padding : stride is 1
        const int n_patches_y = (height - patch_height + 1);
        const int n_patches_x = (width - patch_width + 1);
        const int n_patches = n_patches_x * n_patches_y;

        const int literals_features_per_patch = (patch_width * patch_height) * channels;
        const int position_emb_x_size = (width - patch_width);
        const int position_emb_y_size = (height - patch_height);

        const int literals_per_patch = literals_features_per_patch + position_emb_x_size + position_emb_y_size;
        int total_mem = n_examples * n_patches * literals_per_patch * sizeof(uint8_t);
        //uint8_t* dst_root = (uint8_t*)aligned_alloc(32, total_mem);
        //uint8_t* dst_root = (uint8_t*)malloc(total_mem);
        uint8_t* dst_root = reinterpret_cast<uint8_t*>(safe_aligned_alloc(32, total_mem));
        
        for(int example_index = 0; example_index < n_examples; ++ example_index)
        {
            uint8_t* example = &src[ example_index * (width * height * channels)];
            uint8_t* dst = &dst_root[ example_index * (n_patches * literals_per_patch) ];

            int patch_index = 0;
            for(int py = 0; py < n_patches_y; ++py)
            {
                for(int px = 0; px < n_patches_x; ++px)
                {                                     
                    uint8_t* patch_dst = dst + (patch_index * literals_per_patch);

                    cumlative_encode(patch_dst, py, position_emb_y_size);
                    patch_dst += position_emb_y_size;
                    
                    cumlative_encode(patch_dst, px, position_emb_x_size);                
                    patch_dst += position_emb_x_size;

                    for(int y = 0; y < patch_height; ++y)
                    {
                        uint8_t* patch_row_start = example + ((py+y)*width*channels) + (px * channels);
                        std::memcpy(patch_dst, patch_row_start, patch_width * channels);
                        patch_dst += patch_width * channels;
                    }
                    patch_index += 1;
                }
            }
        }

        // wrapper for return 
        pybind11::capsule free_when_done(dst_root, [](void *f) {
            uint8_t* array_mem = reinterpret_cast<uint8_t *>(f);
            free(array_mem);
        });

        return pybind11::array_t<uint8_t>(
            {n_examples, n_patches, literals_per_patch}, // shape
            dst_root, // the data pointer
            free_when_done);
    }


}; // namespace green_tsetin

#endif // #define __INPUT_BLOCK_HPP_
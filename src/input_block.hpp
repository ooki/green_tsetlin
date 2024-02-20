#ifndef __INPUT_BLOCK_HPP_
#define __INPUT_BLOCK_HPP_


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdint.h>

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
                {                                            
                    m_current_label = &m_labels[index*m_num_labels_per_example];
                }
                    
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

            virtual bool is_label_block() const
            {
                return m_labels != nullptr;
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

}; // namespace green_tsetin

#endif // #define __INPUT_BLOCK_HPP_
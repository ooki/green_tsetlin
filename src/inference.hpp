#ifndef _INFERENCE_HPP_
#define _INFERENCE_HPP_

#include <iostream>
#include <vector>
#include <vector>
#include <cstdlib>
#include <unordered_set>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace green_tsetlin
{
    typedef std::vector<uint32_t> InferenceRule;    
    typedef std::vector<int32_t> RuleWeights;

    template <typename _ExampleType, bool calculate_feature_importance, bool calculate_literal_importance, bool exclude_negative_clauses>
    class Inference
    {
        public:
            typedef _ExampleType example_type;

            Inference(int num_literals, int num_classes, int num_explanations)
            {
                m_num_literals = num_literals;
                m_num_classes = num_classes;
                m_num_explanations = num_explanations;

                m_votes.resize(m_num_classes, 0);

                if(calculate_literal_importance)
                    m_importance_literals.resize(m_num_literals*2, 0.0);                

                if(calculate_feature_importance)                                 
                    m_importance_features.resize(num_explanations, 0.0);                                   
                
                // change to aligned malloc?
                m_example = new example_type[m_num_literals*2];
                //m_clause_features = new uint32_t[]                
            }

            virtual ~Inference()
            {
                if(m_example != nullptr)
                {
                    delete[] m_example;
                    m_example = nullptr;
                }
            }

            void set_empty_class_output(uint32_t empty_class)
            {
                m_empty_class = empty_class;
            }

            uint32_t get_empty_class_output()
            {
                return m_empty_class;
            }


            void set_rules(std::vector<InferenceRule>& l, std::vector<RuleWeights>& w)
            {
                m_rules = l;                
                m_weights = w;
            }

            void set_features(std::vector<InferenceRule>& f)
            {
                if(calculate_feature_importance)
                {
                    if(f.size() != m_num_explanations)
                        throw std::runtime_error("Number of features much match number of explanations set.");   

                    m_features = f;
                }
                else
                {
                    throw std::runtime_error("Cannot set features while note requesting feature importance.");   
                }
            }

            uint32_t predict_npy(pybind11::array_t<example_type> examples)
            {
                pybind11::buffer_info buffer_info = examples.request();
                std::vector<ssize_t> shape = buffer_info.shape;

                example_type* example_ptr = static_cast<example_type*>(buffer_info.ptr);
                
                uint32_t y_hat = predict(example_ptr);
                return y_hat;
            }

            void calculate_explanations(uint32_t target_class)
            {
                calculate_importance(target_class); 
            }


            pybind11::array_t<int32_t> get_votes_npy() 
            {
                return pybind11::cast(m_votes);
            }

            pybind11::array_t<int> get_active_clauses_npy()
            {
                return pybind11::cast(m_active_clauses);
            }

            pybind11::array_t<int32_t> get_literal_importance_npy()
            {                               
                return pybind11::cast(m_importance_literals);
            }

            pybind11::array_t<int32_t> get_feature_importance_npy()
            {   
                return pybind11::cast(m_importance_features);
            }

            uint32_t predict(example_type* example)
            {                
                std::fill(m_votes.begin(), m_votes.end(), 0);
                m_active_clauses.clear();
                // copy -> rewrite to use vectors

                memcpy(m_example, example, m_num_literals * sizeof(example_type));
                for(int i = 0; i < m_num_literals; i++)
                    m_example[m_num_literals+i] = !m_example[i];

                for(uint32_t clause_k = 0; clause_k < m_rules.size(); clause_k++)
                {                    
                    for(uint32_t lit_index : m_rules[clause_k])
                    {
                        if(m_example[lit_index] == 0)       
                        {
                            //std::cout << "predict - exit: clause: " << clause_k << " l_i: " << lit_index << " x[.] == 0 (" << m_example[lit_index] << ")" << std::endl;
                            goto end_of_clause;                        
                        }
                    }

                    m_active_clauses.push_back(clause_k);

                    for(int i = 0; i < m_num_classes; i++)
                        m_votes[i] += m_weights[clause_k][i];

                    end_of_clause:;
                }

                if(m_active_clauses.size() == 0)
                    return m_empty_class;

                int best_k = 0;
                int best_v = m_votes[0];
                for(int i = 1; i < m_num_classes; ++i)
                {
                    if(m_votes[i] > best_v)
                    {
                        best_k = i;
                        best_v = m_votes[i];
                    }
                }

                return best_k;
            }

            void calculate_importance(uint32_t y)
            {
                if(calculate_literal_importance)
                    std::fill(m_importance_literals.begin(), m_importance_literals.end(), 0);

                if(calculate_feature_importance)
                    std::fill(m_importance_features.begin(), m_importance_features.end(), 0);


                for(auto it = m_active_clauses.begin(); it != m_active_clauses.end(); it++)
                {                                         
                    const int clause_k = *it;
                    int32_t w_i = m_weights[clause_k][y];
                    
                    if(exclude_negative_clauses)
                    {
                        if(w_i < 1) // only count positive clauses (that count towards the class)
                            continue;
                    }

                    if(calculate_feature_importance)
                    {
                        for(uint32_t feature_index : m_features[clause_k])
                        {
                            m_importance_features[feature_index] += w_i;
                        }

                    }

                    if(calculate_literal_importance)
                    {
                        for(uint32_t literal_k : m_rules[clause_k])
                        {  
                            m_importance_literals[literal_k] += w_i;
                        }
                    }
                }
            }



        protected:
            int m_num_literals = 0;
            int m_num_clauses = 0;
            int m_num_classes = 0;
            size_t m_num_explanations = 0;
            uint32_t m_empty_class = 0;

            size_t explanation_cache_counter = 0;

            std::vector<InferenceRule>  m_rules;
            std::vector<InferenceRule>  m_features;

            std::vector<RuleWeights> m_weights;
            std::vector<int>         m_active_clauses;
            std::vector<int32_t>     m_importance_literals;
            std::vector<int32_t>     m_importance_features;
            std::vector<int32_t>     m_votes;
            example_type* m_example;


    };


}; // namespace green_tsetlin



#endif // #ifndef _INFERENCE_HPP_
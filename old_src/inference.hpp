#ifndef __INFERENCE__HPP_
#define __INFERENCE__HPP_


#include <iostream>
#include <vector>
#include <vector>
#include <cstdlib>
#include <unordered_set>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace green_tsetlin
{
    typedef std::vector<uint32_t> InfRule;
    typedef std::vector<InfRule>  RuleVector;

    typedef std::vector<WeightInt> RuleWeights;


    template <typename _ExampleType, bool calculate_literal_importance>
    class Inference
    {
        public:
            typedef _ExampleType example_type;

            // note that for multi-label the num_classes will be 2x the actual classes, with ON classes first, and OFF afterwards.
            Inference(int num_literals, int num_clauses, int num_classes, int num_features)
            {
                m_num_literals = num_literals;
                m_num_clauses = num_clauses;
                m_num_classes = num_classes;
                m_num_features = num_features;

                m_votes.resize(m_num_classes, 0);
                m_feature_importance.resize(m_num_features, 0.0);

                if(calculate_literal_importance)                 
                    m_literal_importance.resize(m_num_literals*2, 0.0);
                


                // change to aligned malloc?
                m_example = new example_type[m_num_literals*2];
                //m_clause_features = new uint32_t[]                
            }

            void set_empty_class_output(uint32_t empty_class)
            {
                m_empty_class = empty_class;
            }

            uint32_t get_empty_class_output()
            {
                return m_empty_class;
            }

            virtual ~Inference()
            {
                if(m_example != nullptr)
                {
                    delete[] m_example;
                    m_example = nullptr;
                }
            }


            void set_rules_and_features(RuleVector& l, std::vector<RuleWeights>& w, RuleVector& f)
            {
                m_rules = l;
                m_weights = w;  
                m_features = f;
            }


            uint32_t predict_npy(pybind11::array_t<example_type> examples)
            {
                pybind11::buffer_info buffer_info = examples.request();
                std::vector<ssize_t> shape = buffer_info.shape;

                example_type* example_ptr = static_cast<example_type*>(buffer_info.ptr);
                
                uint32_t y_hat = predict(example_ptr);            
                return y_hat;
            }

            pybind11::array_t<uint32_t> predict_multi_npy(pybind11::array_t<example_type> examples)
            {
                pybind11::buffer_info buffer_info = examples.request();
                std::vector<ssize_t> shape = buffer_info.shape;

                example_type* example_ptr = static_cast<example_type*>(buffer_info.ptr);
                return pybind11::cast(predict_multi(example_ptr));
            }

            pybind11::array_t<int32_t> get_votes_npy() 
            {
                return pybind11::cast(m_votes);
            }

            pybind11::array_t<double> calc_local_importance_npy(uint32_t target_class, bool normalize)
            {   
                calculate_importance_score(target_class, normalize);             
                return pybind11::cast(m_feature_importance);
            }

            pybind11::array_t<double> get_cached_literal_importance_npy()
            {
                return pybind11::cast(m_literal_importance);
            }

            pybind11::array_t<int> get_active_clauses_npy()
            {
                return pybind11::cast(m_active_clauses);
            }

            pybind11::array_t<uint32_t> get_rule_by_literals_npy(int k)
            {
                return pybind11::cast(m_rules[k]);
            }


            pybind11::array_t<double>  calculate_global_importance(int y, bool normalize)
            {
                m_active_clauses.resize(m_num_clauses);
                std::iota(m_active_clauses.begin(), m_active_clauses.end(), 0); // set all clauses to active => [ fill with 0 -> m_num_clauses-1 ]
                calculate_importance_score(y, normalize);

                return pybind11::cast(m_feature_importance);
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
                            goto end_of_clause;                        
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

            std::vector<uint32_t> predict_multi(example_type* example)
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
                            goto end_of_clause;                        
                    }

                    m_active_clauses.push_back(clause_k);

                    for(int i = 0; i < m_num_classes; i++)
                        m_votes[i] += m_weights[clause_k][i];

                    end_of_clause:;
                }                

                
                int num_output_classes = m_num_classes / 2;
                std::vector<uint32_t> multi_predict(num_output_classes, 0);
                for(int i = 0; i < num_output_classes; ++i)
                { 
                    if(m_votes[i] >= m_votes[i + num_output_classes])                    
                        multi_predict[i] = 1;                    
                }

                return multi_predict;
            }

        protected:

            void calculate_importance_score(uint32_t y, bool normalize)
            {
                std::fill(m_feature_importance.begin(), m_feature_importance.end(), 0);

                if(calculate_literal_importance)
                    std::fill(m_literal_importance.begin(), m_literal_importance.end(), 0);

                double z_f = 0.0;
                double z_l = 0.0;
                for(auto it = m_active_clauses.begin(); it != m_active_clauses.end(); it++)
                {                                         
                    const int clause_k = *it;
                    int16_t w_i = m_weights[clause_k][y];
                    
                    if(w_i < 1) // only count positive clauses (that count towards the class)
                        continue;

                    double w = (double)w_i;                        
                    for(uint32_t feature_index : m_features[clause_k])
                    {
                        m_feature_importance[feature_index] += w;
                        z_f += w;
                    }
                    
                    if(calculate_literal_importance)
                    {
                        for(uint32_t literal_k : m_rules[clause_k])
                        {  
                            m_literal_importance[literal_k] += w;     
                            z_l += w;
                        }
                    }
                }

                if(normalize)
                {
                    z_f = std::max(z_f, 1.0);

                    for(int feature_k = 0; feature_k < m_num_features; feature_k++)
                        m_feature_importance[feature_k] /= z_f;

                    if(calculate_literal_importance)
                    {
                        z_l = std::max(z_l, 1.0);
                        for(int literal_k = 0; literal_k < m_num_literals; literal_k++)
                            m_literal_importance[literal_k] /=  z_l;                    
                    }
                }
            }


            int m_num_literals = 0;
            int m_num_clauses = 0;
            int m_num_classes = 0;
            int m_num_features = 0;
            uint32_t m_empty_class = 0;

            RuleVector  m_rules;
            RuleVector  m_features;

            std::vector<RuleWeights> m_weights;
            std::vector<int>         m_active_clauses;

            std::vector<double>     m_feature_importance;
            std::vector<double>     m_literal_importance;
            
            std::vector<int32_t> m_votes;
            example_type* m_example;
    };

}; // namespace green_tsetlin






#endif // #ifndef __INFERENCE__HPP_







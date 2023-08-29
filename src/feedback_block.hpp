#ifndef __FEEDBACK_BLOCK_HPP_
#define __FEEDBACK_BLOCK_HPP_


#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <thread>
#include <mutex>
#include <random>
#include <vector>
#include <array>

#include <random_generator.hpp>

#include <gt_common.hpp>

namespace green_tsetlin
{


    class FeedbackBlock
    {
        public:
            FeedbackBlock(int num_classes, double threshold, int seed)
            {
                m_num_classes = num_classes;
                m_threshold = threshold;

                m_update_probability.resize(m_num_classes, 0.0);      
                m_votes.resize(m_num_classes, 0);
                
                if(seed == 0)
                    seed = std::random_device()();                
                m_rng.seed(seed);
            }

            FeedbackBlock(int num_classes, double threshold)
                : FeedbackBlock(num_classes, threshold, 0)
            {}

            virtual ~FeedbackBlock() {}


            uint32_t get_positive_class() const
            {
                return m_positive_class;
            }

            uint32_t get_negative_class() const
            {
                return m_negative_class;
            }

            double get_positive_update_probability() const 
            {
                return m_update_prob_positive;
            }

            double get_negative_update_probability() const 
            {
                return m_update_prob_negative;
            }

            void reset_train_predict_counter()
            {
                m_correct_train_predict = 0.0;
                m_total_train_predict = 0.0;
            }

            double get_train_accuracy() const
            {
                if(m_total_train_predict > 0.0)                
                    return m_correct_train_predict / m_total_train_predict;
                
                return 0.0;
            }

            virtual void process(const uint32_t* labels)
            {
                
                std::scoped_lock lock(m_votes_lock);

                // TODO: support multi label 
                uint32_t positive_class = labels[0];                
                m_positive_class = positive_class;
                
                WeightInt most_votes = m_votes[0];
                uint32_t predicted_class = 0;

                // calcuate update probabilies and find the negative target using negative focused sampling.
                for(int class_k = 0; class_k < m_num_classes; ++class_k)
                {   
                    WeightInt votes = m_votes[class_k];
                    if(votes > most_votes)
                    {
                        most_votes = votes;
                        predicted_class = class_k;
                    }

                    double v_clamped = std::clamp(static_cast<double>(votes), -m_threshold, m_threshold);
                    m_update_probability[class_k] = ( (m_threshold + v_clamped) / (2*m_threshold) ) + 1e-30; // 1e-30 is used as epsilon                 
                    
                    //if(class_k != m_positive_class)
                    //    std::cout << "\t update prob of class " << class_k << " is: " << m_update_probability[class_k] << " from:" << m_votes[class_k] << std::endl;
                }

                m_total_train_predict += 1.0;
                if(predicted_class == positive_class)                
                {
                    m_correct_train_predict += 1.0;
                }
                    
                
                m_update_probability[m_positive_class] = 0.0; // dont select the positive class as negative class
                std::discrete_distribution<int> weighted_sampler(m_update_probability.begin(), m_update_probability.end());
                m_negative_class = weighted_sampler(m_rng);
                m_update_prob_negative = m_update_probability[m_negative_class];
                

                // get the positive update prob
                double v_clamped_pos = std::clamp(static_cast<double>(m_votes[m_positive_class]), -m_threshold, m_threshold);
                m_update_prob_positive =  (m_threshold - v_clamped_pos) / (2*m_threshold);

                
                //std::cout << "\t update prob of class " << m_positive_class << " is: " << m_update_probability[m_positive_class] << " from:" << m_votes[m_positive_class] << std::endl;

                //std::cout << "neg class is:" << m_negative_class << std::endl;
                //std::cout << "pos class is " << m_positive_class << std::endl;

            }

            virtual int predict() const
            {
                // TODO: determine if we should use clipped votes or unclipped votes here!
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

            virtual std::vector<int> predict_multi() const
            {
                int best_k = 0;
                int best_v = m_votes[0];
                
                std::vector<int> o;
                o.resize(m_num_classes, 0);

                for(int i = 1; i < m_num_classes; ++i)
                {
                    if(m_votes[i] > best_v)
                    {
                        best_k = i;
                        best_v = m_votes[i];
                    }
                }
                
                o[best_k] = 1;
                return o;
            }

            std::vector<double> get_update_probabilities() const
            {
                //std::fill(m_votes.begin(), m_votes.end(), 1.0);
                std::scoped_lock lock(m_votes_lock);
                return m_update_probability;
            }

            void register_votes(WeightInt* votes)
            {
                std::scoped_lock lock(m_votes_lock);
                
                //std::cout << "register vote" << std::endl;
                for(int i = 0; i < m_num_classes; ++i)
                {
                    //std::cout << "register vote: " << i << " adding:" <<  votes[i] << std::endl;
                    m_votes[i] += votes[i];
                }
            }

            void reset()
            {
                std::scoped_lock lock(m_votes_lock);
                std::fill(m_votes.begin(), m_votes.end(), 0);
            }

            int get_number_of_classes() const
            {
                return m_num_classes;
            }

        protected:
            int m_num_classes = -42;       
            double m_threshold = -1337.0;

            // random numbers : TODO fix so it is handled fast and in a more centralized way            
            std::default_random_engine m_rng;

            //Wyhash64 m_rng;

            std::vector<WeightInt>  m_votes;
            std::vector<double>  m_update_probability;

            uint32_t m_positive_class = 0;
            uint32_t m_negative_class = 0;            
            double m_update_prob_positive = 0.0;
            double m_update_prob_negative = 0.0;


            double m_correct_train_predict = 0.0;
            double m_total_train_predict = 0.0;

            mutable std::mutex m_votes_lock;
    };

    
    class FeedbackBlockMultiLabel : public FeedbackBlock
    {
        public:
            FeedbackBlockMultiLabel(int num_classes, double threshold, int seed)                
                : FeedbackBlock(num_classes, threshold, seed)
            {                

                m_votes.resize(num_classes * 2, 0);    // since we now have both + and - of each class.                         
                                                       // layout: [0 - n_classes] : ON
                                                       //         [n_classes - (n_clases*2)]  : OFF
            }


            virtual void process(const uint32_t* labels)
            {
                std::scoped_lock lock(m_votes_lock);


                std::uniform_int_distribution<int>  c_gen(0, m_num_classes - 1);                
                m_positive_class = c_gen(m_rng);               
                m_negative_class = c_gen(m_rng);               
                while(m_negative_class == m_positive_class)
                    m_negative_class = c_gen(m_rng);

                if(labels[m_positive_class] == 0)
                    m_positive_class += m_num_classes;

                if(labels[m_negative_class] == 1)
                    m_negative_class += m_num_classes;
                                    

                WeightInt votes = m_votes[m_positive_class];
                double v_clamped_pos = std::clamp(static_cast<double>(m_votes[m_positive_class]), -m_threshold, m_threshold);
                m_update_prob_positive =  (m_threshold - v_clamped_pos) / (2*m_threshold);


                votes = m_votes[m_negative_class];                
                double v_clamped = std::clamp(static_cast<double>(votes), -m_threshold, m_threshold);
                m_update_prob_negative = ( (m_threshold + v_clamped) / (2*m_threshold) );


                for(int class_k = 0; class_k < m_num_classes; class_k++)
                {
                    m_total_train_predict += 1.0;

                    if(labels[class_k] == 1)
                    {
                       if(m_votes[class_k] >= m_votes[class_k + m_num_classes])
                            m_correct_train_predict += 1.0;
                    }
                    else // label is 0 
                    {
                        if(m_votes[class_k] < m_votes[class_k + m_num_classes])
                            m_correct_train_predict += 1.0;
                    }
                }

            }

            virtual std::vector<int> predict_multi() const
            {               
                std::vector<int> preds(m_num_classes, 0);
                for(int i = 0; i < m_num_classes; ++i)
                {                    
                    if(m_votes[i] >= m_votes[i+m_num_classes])
                        preds[i] = 1;                          
                }

                return preds;
            }      
    };


}; // namespace green_tsetlin

#endif // #ifndef __FEEDBACK_BLOCK_HPP_

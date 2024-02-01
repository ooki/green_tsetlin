#ifndef _CIRCULAR_BUFFER_HPP_
#define _CIRCULAR_BUFFER_HPP_


#include <vector>
#include <unordered_set>

namespace green_tsetlin
{
    template<typename _T>
    class CircularBuffer
    {
        public:
            
            CircularBuffer()
            {
                m_size = 0;
                m_head = 0;
            }

            CircularBuffer(unsigned int size)
            {
                init(size);
            }

            void init(unsigned int size)
            {
                m_members.reserve(size);
                m_data.resize(size);
                m_capacity = size;
                
                m_size = 0;
                m_head = 0;
            }

            bool contains(_T value)
            {
                if(m_members.find(value) == m_members.end())
                    return false;
                else
                    return true;
            }

            void push(_T value)
            {
                if(m_members.find(value) != m_members.end())
                {
                    return; // already in buffer
                }

                if(m_size == m_capacity)
                    m_members.erase(m_data[m_head]);

                m_members.emplace(value);
                m_data[m_head] = value;                
                m_head++;
                
                if(m_size < m_capacity)
                    m_size++;
                else
                    m_head = m_head % m_size;
                
            }

            _T& operator[](unsigned int index)
            {
                return m_data[index];
            }

            unsigned int size() const
            {
                return m_size;
            }

            operator std::vector<int>() const
            {
                return std::vector<int>(m_data.begin(), m_data.begin()+m_size);
            }

            typedef _T value_type;



        private:
            std::vector<_T> m_data;
            std::unordered_set<_T> m_members;
            unsigned int m_head;
            unsigned int m_capacity;
            unsigned int m_size;
    };

};




#endif // #ifndef _CIRCULAR_BUFFER_HPP_
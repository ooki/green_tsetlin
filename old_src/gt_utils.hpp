#ifndef _GT_UTILS_HPP_
#define _GT_UTILS_HPP_

#include <istream>
#include <ostream>
#include <streambuf>

namespace green_tsetlin
{
    /*

    
    struct IMemBuf: std::streambuf
    {
        IMemBuf(const char* base, size_t size)
        {
            char* p(const_cast<char*>(base));
            this->setg(p, p, p + size);
        }
    };

    struct IMemStream: virtual IMemBuf, std::istream
    {
            IMemStream(const char* mem, size_t size) :
                IMemBuf(mem, size),
                std::istream(static_cast<std::streambuf*>(this))
            {
            }
    };

    struct OMemBuf: std::streambuf
    {
        OMemBuf(char* base, size_t size)
        {
            this->setp(base, base + size);
        }
    };

    struct OMemStream: virtual OMemBuf, std::ostream
    {
        OMemStream(char* base, size_t size) :
            OMemBuf(mem, size),
            std::ostream(static_cast<std::streambuf*>(this))
        {
        }
    };
    */

}; // namespace green_tsetlin


#endif // #ifndef _GT_UTILS_HPP_
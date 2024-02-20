#include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <pybind11/stl_bind.h>



#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


// #include <clause_block.hpp>
// #include <executor.hpp>
#include <input_block.hpp>


namespace py = pybind11;
namespace gt = green_tsetlin;



typedef typename gt::DenseInputBlock<uint8_t>   DenseInputBlock8u;


int get_hello_word()
{
    return 42;
}


PYBIND11_MODULE(green_tsetlin_core, m) {

    m.doc() = R"pbdoc(
        green_tsetlin_core
        -----------------------

        .. currentmodule:: green_tsetlin_core

        .. autosummary::
           :toctree: _generate

           A impl. of the Tsetlin Machine
    )pbdoc";

    // hw info
    // m.def("get_recommended_number_of_threads", &gt::get_recommended_number_of_threads);
    m.def("get_hello_word", get_hello_word);

    
    py::class_<gt::InputBlock>(m, "InputBlock")
        .def("prepare_example", &gt::InputBlock::prepare_example)
        .def("is_multi_label", &gt::InputBlock::is_multi_label)        
        .def("get_num_labels_per_example", &gt::InputBlock::get_num_labels_per_example)        
        .def("get_number_of_examples", &gt::InputBlock::get_number_of_examples)
    ;

    py::class_<DenseInputBlock8u, gt::InputBlock>(m, "DenseInputBlock")
        .def(py::init<int>())
        .def("set_data", &DenseInputBlock8u::set_data)
    ;
    



    #ifdef VERSION_INFO
        m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
    #else
        m.attr("__version__") = "dev";
    #endif
}

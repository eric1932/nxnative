// Experimental CPython implementation of dmsh_rewrite.precise. No game data.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cstring>
#include <limits>
#include <new>
#include <string_view>
#include <vector>

template <typename Char>
static bool literal_at(std::basic_string_view<Char> src, size_t pos, const char* text) {
    for (; *text; ++text, ++pos) {
        if (pos >= src.size() || src[pos] != static_cast<Char>(*text)) return false;
    }
    return true;
}

template <typename Char>
static PyObject* rewrite(PyObject* source) {
    const auto length = PyUnicode_GET_LENGTH(source);
    const auto* data = static_cast<const Char*>(PyUnicode_DATA(source));
    const std::basic_string_view<Char> src(data, static_cast<size_t>(length));
    const Char prefix[] = {'f', 'a', 's', 't', ':', ':'};
    const std::basic_string_view<Char> needle(prefix, 6);
    std::vector<size_t> hits;
    for (size_t pos = src.find(needle); pos != src.npos; pos = src.find(needle, pos + 6)) {
        // Python's Unicode \w is alphanumeric or underscore, not C's locale-dependent isalnum.
        if (pos && (src[pos - 1] == '_' || Py_UNICODE_ISALNUM(src[pos - 1]))) continue;
        if (literal_at(src, pos + 6, "min(") || literal_at(src, pos + 6, "max(") ||
            literal_at(src, pos + 6, "clamp(")) hits.push_back(pos);
    }
    if (hits.empty()) return Py_NewRef(source);
    if (hits.size() > static_cast<size_t>((PY_SSIZE_T_MAX - length) / 3)) return PyErr_NoMemory();
    PyObject* output = PyUnicode_New(length + static_cast<Py_ssize_t>(hits.size()) * 3,
                                     PyUnicode_MAX_CHAR_VALUE(source));
    if (!output) return nullptr;
    auto* dst = static_cast<Char*>(PyUnicode_DATA(output));
    size_t read = 0, write = 0;
    for (size_t pos : hits) {
        const auto count = pos - read;
        std::memcpy(dst + write, data + read, count * sizeof(Char));
        write += count;
        for (char ch : std::string_view("precise::")) dst[write++] = static_cast<Char>(ch);
        read = pos + 6;
    }
    std::memcpy(dst + write, data + read, (src.size() - read) * sizeof(Char));
    return output;
}

static PyObject* precise(PyObject*, PyObject* source) {
    if (!PyUnicode_Check(source)) {
        PyErr_SetString(PyExc_TypeError, "precise expects str");
        return nullptr;
    }
    try {
        switch (PyUnicode_KIND(source)) {
            case PyUnicode_1BYTE_KIND: return rewrite<Py_UCS1>(source);
            case PyUnicode_2BYTE_KIND: return rewrite<Py_UCS2>(source);
            case PyUnicode_4BYTE_KIND: return rewrite<Py_UCS4>(source);
        }
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    }
    PyErr_SetString(PyExc_RuntimeError, "unsupported Unicode representation");
    return nullptr;
}

static PyMethodDef methods[] = {
    {"precise", precise, METH_O, "Rewrite fast min/max/clamp exactly as the upstream regex."},
    {nullptr, nullptr, 0, nullptr}
};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_precise_native", nullptr, -1,
                             methods, nullptr, nullptr, nullptr, nullptr};
PyMODINIT_FUNC PyInit__precise_native() { return PyModule_Create(&module); }

#include <float.h> // DBL_MAX
#include <limits> // numeric_limits

template<typename T>
inline bool is_infinite( const T &value )
{
    // Since we're a template, it's wise to use std::numeric_limits<T>
    //
    // Note: std::numeric_limits<T>::min() behaves like DBL_MIN, and is the smallest absolute value possible.
    //
 
    T max_value = std::numeric_limits<T>::max();
    T min_value = - max_value;
 
    return !( min_value <= value && value <= max_value );
}
 
template<typename T>
inline bool is_nan( const T &value )
{
    // True if NAN
    return value != value;
}
 
template<typename T>
inline bool is_valid( const T &value )
{
    return !is_infinite(value) && !is_nan(value);
}


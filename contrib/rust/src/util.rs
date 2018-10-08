/// The `Wrap` trait provides common interfaces for a raw pointer.
pub trait Wrap<T>: Drop {
    /// Creates an object from a raw pointer.
    ///
    /// The caller must specifies whether ownership of the value that the pointer references is
    /// transferred or not.
    fn from_raw(ptr: *mut T, owned: bool) -> Self
    where
        Self: Sized;

    /// Returns the raw pointer.
    fn as_ptr(&self) -> *const T;

    /// Returns the mutable raw pointer.
    fn as_mut_ptr(&mut self) -> *mut T;

    /// Returns whether the object has ownership of what the raw pointer references.
    fn is_owned(&self) -> bool;
}

macro_rules! impl_wrap {
    ($name:ident, $type:ident) => {
        impl Wrap<dynet_sys::$type> for $name {
            #[inline(always)]
            fn from_raw(ptr: *mut dynet_sys::$type, owned: bool) -> Self {
                $name {
                    inner: NonNull::new(ptr).expect("pointer must not be null"),
                    owned,
                }
            }

            #[inline(always)]
            fn as_ptr(&self) -> *const dynet_sys::$type {
                self.inner.as_ptr()
            }

            #[inline(always)]
            fn as_mut_ptr(&mut self) -> *mut dynet_sys::$type {
                self.inner.as_ptr()
            }

            #[inline(always)]
            fn is_owned(&self) -> bool {
                self.owned
            }
        }
    };
}

macro_rules! impl_wrap_owned {
    ($name:ident, $type:ident) => {
        impl Wrap<dynet_sys::$type> for $name {
            #[inline(always)]
            fn from_raw(ptr: *mut dynet_sys::$type, _owned: bool) -> Self {
                $name {
                    inner: NonNull::new(ptr).expect("pointer must not be null"),
                }
            }

            #[inline(always)]
            fn as_ptr(&self) -> *const dynet_sys::$type {
                self.inner.as_ptr()
            }

            #[inline(always)]
            fn as_mut_ptr(&mut self) -> *mut dynet_sys::$type {
                self.inner.as_ptr()
            }

            #[inline(always)]
            fn is_owned(&self) -> bool {
                true
            }
        }
    };
}

macro_rules! impl_drop {
    ($name:ident, $call:ident) => {
        impl Drop for $name {
            fn drop(&mut self) {
                if self.is_owned() {
                    unsafe {
                        check_api_status!(dynet_sys::$call(self.as_mut_ptr()));
                    }
                }
            }
        }
    };
}

use std::ffi::CString;
use std::fmt::{self, Debug, Display, Formatter};
use std::{env, mem, ptr, result};

use backtrace::Backtrace;
use dynet_sys;
use libc::c_uint;

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Copy, Clone)]
pub(crate) enum Code {
    Ok,
    Error,
    UnrecognizedEnumValue(c_uint),
}

impl Code {
    pub fn from_int(value: c_uint) -> Self {
        match value {
            0 => Code::Ok,
            4294967295 => Code::Error,
            c => Code::UnrecognizedEnumValue(c),
        }
    }

    pub fn to_int(&self) -> c_uint {
        match self {
            &Code::UnrecognizedEnumValue(c) => c,
            &Code::Ok => 0,
            &Code::Error => 4294967295,
        }
    }

    #[allow(dead_code)]
    fn to_c(&self) -> dynet_sys::DYNET_C_STATUS {
        unsafe { mem::transmute(self.to_int()) }
    }

    #[allow(dead_code)]
    fn from_c(value: dynet_sys::DYNET_C_STATUS) -> Self {
        Self::from_int(value)
    }

    pub fn is_ok(value: c_uint) -> bool {
        match value {
            0 => true,
            _ => false,
        }
    }
}

impl Display for Code {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            &Code::Ok => f.write_str("Ok"),
            &Code::Error => f.write_str("Error"),
            &Code::UnrecognizedEnumValue(c) => write!(f, "UnrecognizedEnumValue({})", c),
        }
    }
}

pub(crate) struct Status {
    code: Code,
    message: String,
    trace: Option<Backtrace>,
}

impl Status {
    fn new(code: Code, message: String, trace: Option<Backtrace>) -> Self {
        Status {
            code,
            message,
            trace,
        }
    }

    pub fn code(&self) -> Code {
        self.code
    }

    pub fn message(&self) -> &str {
        self.message.as_str()
    }
}

impl Display for Status {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        try!(write!(
            f,
            "Code: \"{}({})\", Message: \"{}\"\n",
            self.code(),
            self.code().to_int(),
            self.message()
        ));
        Ok(())
    }
}

impl Debug for Status {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        try!(write!(f, "Status: {{"));
        try!(write!(
            f,
            "code: \"{}({})\", message: \"{}\"",
            self.code(),
            self.code().to_int(),
            self.message()
        ));
        if let Some(ref trace) = self.trace {
            try!(write!(f, ", backtrace: \"\n{:?}\n\"", trace));
        }
        try!(write!(f, "}}"));
        Ok(())
    }
}

pub(crate) trait ApiResult<T, E> {
    fn from_api_status(status: c_uint, ok_val: T) -> result::Result<T, E>;
}

impl<T> ApiResult<T, Status> for result::Result<T, Status> {
    fn from_api_status(status: c_uint, ok_val: T) -> Self {
        let code = Code::from_int(status);
        match code {
            Code::Ok => Ok(ok_val),
            _ => unsafe {
                let trace = match env::var_os("RUST_BACKTRACE") {
                    Some(ref val) if val != "0" => Some(Backtrace::new()),
                    _ => None,
                };
                let mut size: usize = 0;
                let s = dynet_sys::dynetGetMessage(ptr::null_mut(), &mut size);
                assert!(Code::is_ok(s));
                let buffer = CString::new(vec![b'0'; size]).unwrap().into_raw();
                let s = dynet_sys::dynetGetMessage(buffer, &mut size);
                assert!(Code::is_ok(s));
                let message = CString::from_raw(buffer).into_string().unwrap();
                Err(Status::new(code, message, trace))
            },
        }
    }
}

pub(crate) type Result<T> = result::Result<T, Status>;

macro_rules! check_api_status {
    ($status:expr) => {
        match Result::from_api_status($status, 0) {
            Ok(_) => {}
            Err(s) => {
                panic!("{:?}", s);
            }
        }
    };
}

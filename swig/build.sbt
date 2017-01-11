
name := "dynet_scala_helpers"

scalaVersion := "2.11.8"

unmanagedBase := file( "../build/swig" ).getAbsoluteFile

assemblyJarName in assembly := "dynet_swigJNI_scala.jar"

fork in run := true
javaOptions in run += "-Djava.library.path=../build/swig"

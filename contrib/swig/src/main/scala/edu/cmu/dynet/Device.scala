package edu.cmu.dynet

object Device {
  def apply(str:String): internal.Device = {
    if(str == "" || str == "default") internal.dynet_swig.getDefault_device
    else DeviceManager.getGlobalDevice(str)
  }

  def default(): internal.Device = internal.dynet_swig.getDefault_device

  def available(): Vector[internal.Device] = {
    val tmp = for(l <- 0L until DeviceManager.numDevices()) yield DeviceManager.get(l)
    tmp.toVector
  }
}

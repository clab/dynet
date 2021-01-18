package edu.cmu.dynet

object DeviceManager {
  private[dynet] val dm: internal.DeviceManager = internal.dynet_swig.get_device_manager()

  def add(d: internal.Device): Unit = dm.add(d)

  def get(l: Long): internal.Device = dm.get(l)

  def numDevices(): Long = dm.num_devices()

  def getGlobalDevice(name: String): internal.Device = dm.get_global_device(name)

  def getDefaultDevice: internal.Device = internal.dynet_swig.getDefault_device

  def showMemPoolInfo(): Unit = internal.dynet_swig.show_pool_mem_info()
}

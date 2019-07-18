package edu.cmu

package object dynet {
  implicit class RichDevice(self: internal.Device) {
    def name(): String = self.getName
    def deviceID(): Int = self.getDevice_id
    def resetRNG(seed:Long): Unit = self.reset_rng(seed)
  }
}

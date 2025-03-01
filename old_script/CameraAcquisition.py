import numpy as np 
import cv2
import sys

from ids_peak import ids_peak as peak
from ids_peak_ipl import ids_peak_ipl as ipl
from ids_peak import ids_peak_ipl_extension






m_device = None
m_dataStream = None
m_node_map_remote_device = None
out = None


def open_camera():
  print("connection- camera")
  global m_device, m_node_map_remote_device
  try:
      # Create instance of the device manager
    device_manager = peak.DeviceManager.Instance()
 
      # Update the device manager
    device_manager.Update()
 
      # Return if no device was found
    if device_manager.Devices().empty():
      return False
 
      # open the first openable device in the device manager's device list
    device_count = device_manager.Devices().size()
    for i in range(device_count):
        if device_manager.Devices()[i].IsOpenable():
            m_device = device_manager.Devices()[i].OpenDevice(peak.DeviceAccessType_Control)
 
              # Get NodeMap of the RemoteDevice for all accesses to the GenICam NodeMap tree
            m_node_map_remote_device = m_device.RemoteDevice().NodeMaps()[0]
            min_frame_rate = 0
            max_frame_rate = 50
            inc_frame_rate = 0

            
            # Get frame rate range. All values in fps.
            min_frame_rate = m_node_map_remote_device.FindNode("AcquisitionFrameRate").Minimum()
            max_frame_rate = m_node_map_remote_device.FindNode("AcquisitionFrameRate").Maximum()
            
            if m_node_map_remote_device.FindNode("AcquisitionFrameRate").HasConstantIncrement():
                inc_frame_rate = m_node_map_remote_device.FindNode("AcquisitionFrameRate").Increment()
            else:
                # If there is no increment, it might be useful to choose a suitable increment for a GUI control element (e.g. a slider)
                inc_frame_rate = 0.1
            
            # Get the current frame rate
            frame_rate = m_node_map_remote_device.FindNode("AcquisitionFrameRate").Value()
            
            # Set frame rate to maximum
            m_node_map_remote_device.FindNode("AcquisitionFrameRate").SetValue(max_frame_rate)
        return True
  except Exception as e:
      # ...
    str_error = str(e)
    print("Error by connection camera")
    return False
 
 
def prepare_acquisition():
  print("opening stream")
  global m_dataStream

  try:
    data_streams = m_device.DataStreams()
    if data_streams.empty():
      print("no stream possible")
      # no data streams available
      return False
 
    m_dataStream = m_device.DataStreams()[0].OpenDataStream()
    print("open stream")
 
    return True
  except Exception as e:
      # ...
      str_error = str(e)
      print("Error by prep acquisition")
      return False
 
 
def set_roi(x, y, width, height):
  print("setting ROI")
  try:
      # Get the minimum ROI and set it. After that there are no size restrictions anymore
    x_min = m_node_map_remote_device.FindNode("OffsetX").Minimum()
    y_min = m_node_map_remote_device.FindNode("OffsetY").Minimum()
    w_min = m_node_map_remote_device.FindNode("Width").Minimum()
    h_min = m_node_map_remote_device.FindNode("Height").Minimum()
 
    m_node_map_remote_device.FindNode("OffsetX").SetValue(x_min)
    m_node_map_remote_device.FindNode("OffsetY").SetValue(y_min)
    m_node_map_remote_device.FindNode("Width").SetValue(w_min)
    m_node_map_remote_device.FindNode("Height").SetValue(h_min)
 
      # Get the maximum ROI values
    x_max = m_node_map_remote_device.FindNode("OffsetX").Maximum()
    y_max = m_node_map_remote_device.FindNode("OffsetY").Maximum()
    w_max = m_node_map_remote_device.FindNode("Width").Maximum()
    h_max = m_node_map_remote_device.FindNode("Height").Maximum()
 
    if (x < x_min) or (y < y_min) or (x > x_max) or (y > y_max):
      print("Error x and y values")
      return False
    elif (width < w_min) or (height < h_min) or ((x + width) > w_max) or ((y + height) > h_max):
      print("Error width and height")
      return False
    else:
          # Now, set final AOI
        m_node_map_remote_device.FindNode("OffsetX").SetValue(x)
        m_node_map_remote_device.FindNode("OffsetY").SetValue(y)
        m_node_map_remote_device.FindNode("Width").SetValue(width)
        m_node_map_remote_device.FindNode("Height").SetValue(height)
 
        return True
  except Exception as e:
      # ...
       str_error = str(e)
       print("Error by setting ROI")
       print(str_error)
       return False
 
 
def alloc_and_announce_buffers():
  print("allocating buffers")
  try:
    if m_dataStream:
          # Flush queue and prepare all buffers for revoking
        m_dataStream.Flush(peak.DataStreamFlushMode_DiscardAll)
 
          # Clear all old buffers
        for buffer in m_dataStream.AnnouncedBuffers():
            m_dataStream.RevokeBuffer(buffer)
 
        payload_size = m_node_map_remote_device.FindNode("PayloadSize").Value()
 
          # Get number of minimum required buffers
        num_buffers_min_required = m_dataStream.NumBuffersAnnouncedMinRequired()
 
          # Alloc buffers
        for count in range(num_buffers_min_required):
            buffer = m_dataStream.AllocAndAnnounceBuffer(payload_size)
            m_dataStream.QueueBuffer(buffer)
 
        return True
  except Exception as e:
      # ...
    str_error = str(e)
    print("Error by allocating buffers")
    print(str_error)
    return False
 
 
def start_acquisition():
  print("Start acquisition")

  try:
    m_dataStream.StartAcquisition(peak.AcquisitionStartMode_Default, peak.DataStream.INFINITE_NUMBER)
    m_node_map_remote_device.FindNode("TLParamsLocked").SetValue(1)
    m_node_map_remote_device.FindNode("AcquisitionStart").Execute()
       
    return True
  except Exception as e:
      # ...
      str_error = str(e)
      print(str_error)
      return False

def saving_acquisition():  
  fourcc = cv2.VideoWriter_fourcc('W','M','V','2')
  out = cv2.VideoWriter( "video", fourcc, 50, (1936,  1096))
  while True:
    try:
      
      # Get buffer from device's DataStream. Wait 5000 ms. The buffer is automatically locked until it is queued again.
      buffer = m_dataStream.WaitForFinishedBuffer(5000)

      image = ids_peak_ipl_extension.BufferToImage(buffer)
        
      # Create IDS peak IPL image for debayering and convert it to RGBa8 format
            
      image_processed = image.ConvertTo(ipl.PixelFormatName_BGR8)
      # Queue buffer again
      m_dataStream.QueueBuffer(buffer)
        
      image_python = image_processed.get_numpy_3D()

      frame = image_python
    
      out.write(frame)
      cv2.imshow('videoview',frame)
      
      key = cv2.waitKey(1)
      if key == ord('q'):
        break

      
    except Exception as e:
      # ...
      str_error = str(e)
      print("Error by saving acquisition")
      print(str_error)
      return False

 
def main():
  
  # initialize library
  peak.Library.Initialize()
 
  if not open_camera():
    # error
    sys.exit(-1)
 
  if not prepare_acquisition():
    # error
    sys.exit(-2)
 
  if not alloc_and_announce_buffers():
    # error
    sys.exit(-3)
 
  if not start_acquisition():
    # error
    sys.exit(-4)

  if not saving_acquisition():
    out.release()
    cv2.destroyAllWindows()
    print("oke")
    # error
 
  peak.Library.Close()
  print('executed')
  sys.exit(0)
 
if __name__ == '__main__':
  main()
from ids_peak import ids_peak
 
#  Runs OK when operated on Windows, from VS Code



def main():
    # initialize library
    ids_peak.Library.Initialize()
    
    # create a DeviceManager object
    device_manager = ids_peak.DeviceManager.Instance()

    try:
        # update the DeviceManager
        device_manager.Update()
    
        # exit program if no device was found
        # Devices() list shows available devices
        if device_manager.Devices().empty():
            print("No device found. Exiting Program.")
            return -1

    

        # open the first device
        device = device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
    
        # ... do something with the device here
        print('Should do something')
    
    except Exception as e:
        print("EXCEPTION: " + str(e))
        return -2
    
    finally:
        ids_peak.Library.Close()
    
 
if __name__ == '__main__':

   main()
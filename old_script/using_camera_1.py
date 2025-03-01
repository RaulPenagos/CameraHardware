# include IDS peak
from ids_peak import ids_peak
 
# ...
 
# initialize library
ids_peak.Library.Initialize()



from ids_peak import ids_peak
 
 
def main():
    # Initialize library
    ids_peak.Library.Initialize()
 
    # Create a DeviceManager object
    device_manager = ids_peak.DeviceManager.Instance()
 
    try:
        # Update the DeviceManager
        device_manager.Update()
 
        # Exit program if no device was found
        if device_manager.Devices().empty():
            print("No device found. Exiting Program.")
            return -1
    
        # Open the first device
        device = device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
    
        # ... Do something with the device here
    
    except Exception as e:
        print("EXCEPTION: " + str(e))
        return -2
 
    finally:
        # close library before exiting program 
        ids_peak.Library.Close()
 
 
if __name__ == '__main__':
   main()




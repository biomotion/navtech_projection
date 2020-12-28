#!/usr/bin/env python2
"""
    Notes:
        After using the warping grid the output radar cartesian is defined as as follows where
        X and Y are the `real` world locations of the pixels in metres:
         If 'cart_pixel_width' is odd:
                        +------ Y = -1 * cart_resolution (m)
                        |+----- Y =  0 (m) at centre pixel
                        ||+---- Y =  1 * cart_resolution (m)
                        |||+--- Y =  2 * cart_resolution (m)
                        |||| +- Y =  cart_pixel_width // 2 * cart_resolution (m) (at last pixel)
                        |||| +-----------+
                        vvvv             v
         +---------------+---------------+
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         +---------------+---------------+ <-- X = 0 (m) at centre pixel
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         +---------------+---------------+
         <------------------------------->
             cart_pixel_width (pixels)
         If 'cart_pixel_width' is even:
                        +------ Y = -0.5 * cart_resolution (m)
                        |+----- Y =  0.5 * cart_resolution (m)
                        ||+---- Y =  1.5 * cart_resolution (m)
                        |||+--- Y =  2.5 * cart_resolution (m)
                        |||| +- Y =  (cart_pixel_width / 2 - 0.5) * cart_resolution (m) (at last pixel)
                        |||| +----------+
                        vvvv            v
         +------------------------------+
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         +------------------------------+
         <------------------------------>
             cart_pixel_width (pixels)
"""
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
bridge = CvBridge()
img_pub = None

def radar_polar_to_cartesian(fft_data, azimuths=None, radar_resolution=0.0432,
                             cart_resolution=None, cart_pixel_width=None, interpolate_crossover=True):
    """Convert a polar radar scan to cartesian.
    Args:
        fft_data (np.ndarray): Polar radar power readings
        azimuths (np.ndarray): Rotation for each polar radar azimuth (radians)
        radar_resolution (float): Resolution of the polar radar data (metres per pixel)
        cart_resolution (float): Cartesian resolution (metres per pixel)
        cart_pixel_size (int): Width and height of the returned square cartesian output (pixels). Please see the Notes
            below for a full explanation of how this is used.
        interpolate_crossover (bool, optional): If true interpolates between the end and start  azimuth of the scan. In
            practice a scan before / after should be used but this prevents nan regions in the return cartesian form.

    Returns:
        np.ndarray: Cartesian radar power readings
    """
    if cart_resolution is None:
        cart_resolution = radar_resolution
    if cart_pixel_width is None:
        cart_pixel_width =  2 * fft_data.shape[1] + 1
    if azimuths is None:
        azimuths = np.arange(0, 2*np.pi, np.pi*2/fft_data.shape[0])

    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    coords = np.linspace(-cart_min_range, cart_min_range, cart_pixel_width, dtype=np.float32)
    Y, X = np.meshgrid(coords, -coords)
    sample_range = np.sqrt(Y * Y + X * X)
    sample_angle = np.arctan2(Y, X)
    sample_angle += (sample_angle < 0).astype(np.float32) * 2. * np.pi

    # Interpolate Radar Data Coordinates
    azimuth_step = azimuths[1] - azimuths[0]
    sample_u = (sample_range - radar_resolution / 2) / radar_resolution
    sample_v = (sample_angle - azimuths[0]) / azimuth_step

    # We clip the sample points to the minimum sensor reading range so that we
    # do not have undefined results in the centre of the image. In practice
    # this region is simply undefined.
    sample_u[sample_u < 0] = 0

    if interpolate_crossover:
        fft_data = np.concatenate((fft_data[-1:], fft_data, fft_data[:1]), 0)
        sample_v = sample_v + 1

    polar_to_cart_warp = np.stack((sample_u, sample_v), -1)
    cart_img = cv2.remap(fft_data, polar_to_cart_warp, None, cv2.INTER_LINEAR)
    return cart_img

def img_cb(msg):
    # rospy.loginfo("type=%s"%(str(type(msg))))
    cv_image = bridge.imgmsg_to_cv2(msg)

    cart_img = radar_polar_to_cartesian(cv_image.T)
    
    output_msg = bridge.cv2_to_imgmsg(cart_img)
    output_msg.header = msg.header
    output_msg.header.frame_id = "navtech_optical"
    rospy.loginfo("size=%ix%i"%(output_msg.height, output_msg.width))
    rospy.loginfo("frame = %s" %output_msg.header.frame_id)
    img_pub.publish(output_msg)


if __name__ == "__main__":
    rospy.init_node("image_projector")
    img_pub = rospy.Publisher("/Navtech/Cartesian", Image, queue_size=1)
    img_sub = rospy.Subscriber("/Navtech/Polar", Image, img_cb, queue_size=1)
    rospy.spin()
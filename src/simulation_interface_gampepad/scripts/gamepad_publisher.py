#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32MultiArray, Bool
from threading import Thread
from threading import Lock
from inputs import get_gamepad

end = False
obj = Lock()
name_node = "gamePad"


class XPAD(Thread):  # def class typr thread
    def __init__(self, semaphore):
        Thread.__init__(self)
        self.semaphore = semaphore
        self.LBumper = 0
        self.RBumper = 0
        self.LStickX = 0
        self.LStickY = 0
        self.RStickX = 0
        self.RStickY = 0
        self.LThumb = 0
        self.A = False
        self.Y = False

    def run(self):  # run is a default Thread function
        global end

        while not end:
            for event in get_gamepad():
                with self.semaphore:
                    if event.ev_type == "Key":

                        if event.code == "BTN_THUMBL":
                            self.LThumb = event.state
                        elif event.code == "BTN_TL":
                            self.LBumper = event.state
                        elif event.code == "BTN_TR":
                            self.RBumper = event.state
                        elif event.code == "BTN_WEST":
                            self.Y = bool(event.state)
                            end = True
                        elif event.code == "BTN_SOUTH":
                            self.A = bool(event.state)

                    elif event.ev_type == "Absolute":  # category of analog values

                        if event.code[-1:] == "Z":
                            event.state = event.state << 1  # reduce range from 256 to 512
                        else:
                            event.state = event.state >> 6  # reduce range from 32000 to 512

                        if 40 > event.state > -40:
                            event.state = 0

                        if event.code == "ABS_X":
                            self.LStickX = event.state
                        elif event.code == "ABS_Y":
                            self.LStickY = event.state
                        elif event.code == "ABS_RX":
                            self.RStickX = event.state
                        elif event.code == "ABS_RY":
                            self.RStickY = event.state


def talker():
    global end

    gamePad = XPAD(obj)
    pub = rospy.Publisher('move_command', Float32MultiArray, queue_size=10)
    pub_reset = rospy.Publisher('ur5_simulation/reset', Bool, queue_size=10)
    pub_store = rospy.Publisher('store_command', Bool, queue_size=10)
    rospy.init_node(name_node, anonymous=True)
    rate = rospy.Rate(50)  # 50h

    try:
        gamePad.start()

        while not rospy.is_shutdown():
            with obj:
                mult = 1
                if gamePad.LBumper:
                    mult = 0.5
                elif gamePad.RBumper:
                    mult = 2

                aux_y = gamePad.Y
                gamePad.Y = False

                aux_a = gamePad.A
                gamePad.A = False

                x = gamePad.LStickX / mult
                y = -gamePad.LStickY / mult
                z = -gamePad.RStickY / mult
                aux = [x, y, z, 0, 0, 0] if not gamePad.LThumb else [0, 0, 0, x, y, z]

            mess = Float32MultiArray(data=aux)
            # rospy.loginfo(mess)
            pub.publish(mess)

            if aux_y:
                mess = Bool(data=aux_y)
                pub_store.publish(mess)
                break

            if aux_a:
                mess = Bool(data=aux_a)
                pub_reset.publish(mess)

            rate.sleep()
    finally:
        end = True


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass

#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool, Float32MultiArray
import pandas as pd
import os
import signal
import sys

cwd = os.path.dirname(os.path.abspath(__file__))
full_path_file = ""
name_folder = "simulation_data"

name_node = "sim_ros_interface"
stored_position_data = [{}]
stored_torque_data = [{}]
stored_speed_data = [{}]
stored_tool_position_data = [{}]

stored_data = []

size_position_data = 1
size_torque_data = 1
size_speed_data = 1
size_tool_position_data = 1

exit_sim = False
save_data = False
name_file = "data.xlsx"


def get_parent_dir(current_path, levels=1):
    rc = current_path
    for i in range(levels + 1):
        rc = os.path.dirname(rc)

    return rc


def get_full_folder_data():
    global full_path_file
    aux = os.path.join(get_parent_dir(cwd, levels=3), name_folder)
    try:
        os.mkdir(aux)
    except OSError as _:
        pass

    full_path_file = os.path.join(aux, name_file)

    print(f"Data will be stored in: {full_path_file}")


def callback1(data):
    global size_position_data

    iter_m = int(data.data[0])
    time = int(data.data[1])
    missing_size = iter_m + 1 - size_position_data

    if 0 != missing_size:
        for _ in range(missing_size):
            stored_position_data.append({})
            size_position_data += 1

    stored_position_data[iter_m][str(time)] = data.data[2:]


def callback2(data):
    global size_torque_data

    iter_m = int(data.data[0])
    time = int(data.data[1])
    missing_size = iter_m + 1 - size_torque_data

    if 0 != missing_size:
        for _ in range(missing_size):
            stored_torque_data.append({})
            size_torque_data += 1

    stored_torque_data[iter_m][str(time)] = data.data[2:]


def callback3(data):
    global size_speed_data

    iter_m = int(data.data[0])
    time = int(data.data[1])
    missing_size = iter_m + 1 - size_speed_data

    if 0 != missing_size:
        for _ in range(missing_size):
            stored_speed_data.append({})
            size_speed_data += 1

    stored_speed_data[iter_m][str(time)] = data.data[2:]


def callback5(data):
    global size_tool_position_data

    iter_m = int(data.data[0])
    time = int(data.data[1])
    missing_size = iter_m + 1 - size_tool_position_data

    if 0 != missing_size:
        for _ in range(missing_size):
            stored_tool_position_data.append({})
            size_tool_position_data += 1

    stored_tool_position_data[iter_m][str(time)] = data.data[2:]


def callback4(data):
    global exit_sim
    global save_data
    if data.data:
        exit_sim = True
        save_data = True


def organize_and_store_data():
    last_index = len(stored_speed_data)
    if last_index == len(stored_torque_data) and last_index == len(stored_position_data) and last_index == len(stored_tool_position_data):

        for i in range(last_index - 1):

            for key, item in stored_position_data[i].items():
                aux_dict = {"iteration": i, "time": int(key), "torque": [], "speed": [], "joint_config": item, "tool_position": []}

                if key in stored_torque_data[i]:
                    aux_dict["torque"] = stored_torque_data[i][key]

                if key in stored_speed_data[i]:
                    aux_dict["speed"] = stored_speed_data[i][key]

                if key in stored_tool_position_data[i]:
                    aux_dict["tool_position"] = stored_tool_position_data[i][key]

                stored_data.append(aux_dict)

        df = pd.DataFrame(stored_data)
        df.to_excel(full_path_file)

        print("Saved")
    else:
        print("Error in lengths")


def listener():
    subs = []

    try:
        # In ROS, nodes are uniquely named. If two nodes with the same
        # name are launched, the previous one is kicked off. The
        # anonymous=True flag means that rospy will choose a unique
        # name for our 'listener' node so that multiple listeners can
        # run simultaneously.
        rospy.init_node(name_node, anonymous=True)

        subs = [
            rospy.Subscriber('/ur5_simulation/jointConfig', Float32MultiArray, callback1),
            rospy.Subscriber('/ur5_simulation/torques', Float32MultiArray, callback2),
            rospy.Subscriber('/ur5_simulation/speeds', Float32MultiArray, callback3),
            rospy.Subscriber('store_command', Bool, callback4),
            rospy.Subscriber('/ur5_simulation/ToolPosition', Float32MultiArray, callback5)
        ]

        while not exit_sim:
            pass
    finally:
        for sub in subs:
            sub.unregister()

    if save_data:
        organize_and_store_data()

    # with open('json_data.json', 'w') as outfile:
    #     json.dump(stored_position_data, outfile)

    # spin() simply keeps python from exiting until this node is stopped
    # rospy.spin()


def sigint_handler(_, __):
    global exit_sim
    exit_sim = True
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint_handler)
    get_full_folder_data()
    listener()
    # test()

import math
## =========================== helper functions =============================
def rotate(x, y, theta):
    """
    坐标轴转，物体不转
    formula reference: https://www.cnblogs.com/jiahenhe2/p/10135235.html
    """
    x_n = (
            math.cos(theta * math.pi / 180) * x
            + math.sin(theta * math.pi / 180) * y
    )
    y_n = (
            -math.sin(theta * math.pi / 180) * x
            + math.cos(theta * math.pi / 180) * y
    )
    return x_n, y_n


def rotate2(x, y, theta):
    """
    坐标轴不转，物体转; 坐标点旋转后的点坐标, 逆时针旋转theta
    """
    x_n = (
            math.cos(theta * math.pi / 180) * x
            + math.sin(theta * math.pi / 180) * y
    )
    y_n = (
            -math.sin(theta * math.pi / 180) * x
            + math.cos(theta * math.pi / 180) * y
    )
    return x_n, y_n


def get_distance(AB, vec_OC, AB_length, pixel):
    """
    通过向量叉乘，求点C到线段AB的距离; 通过点乘判断点位置的三种情况，左边、右边和上面
    :param: 两个点AB -> [[].[]]，点C->[]，AB线段长度
    :return:
    formula reference: https://blog.csdn.net/qq_45735851/article/details/114448767
    """
    vec_OA, vec_OB = AB[0], AB[1]
    vec_CA = [vec_OA[0] - vec_OC[0], vec_OA[1] - vec_OC[1]]
    vec_CB = [vec_OB[0] - vec_OC[0], vec_OB[1] - vec_OC[1]]
    vec_AB = [vec_OB[0] - vec_OA[0], vec_OB[1] - vec_OA[1]]

    vec_AC = [-vec_OA[0] + vec_OC[0], -vec_OA[1] + vec_OC[1]]
    vec_BC = [-vec_OB[0] + vec_OC[0], -vec_OB[1] + vec_OC[1]]

    if pixel:
        if dot(vec_AB, vec_AC) < 0:
            d = distance_2points(vec_AC)
        elif dot(vec_AB, vec_BC) > 0:
            d = distance_2points(vec_BC)
        else:
            d = math.ceil(cross(vec_CA, vec_CB) / AB_length)
    else:
        d = math.ceil(cross(vec_CA, vec_CB) / AB_length)
    return d


def dot(vec_1, vec_2):
    """
    计算点乘，vec_1, vec_2都为向量
    """
    return vec_1[0] * vec_2[0] + vec_1[1] * vec_2[1]


def cross(vec_1, vec_2):
    """
    计算叉积，vec_1, vec_2都为向量
    """
    return vec_1[0] * vec_2[1] - vec_1[1] * vec_2[0]


def distance_2points(vec):
    """
    计算两个点直接距离，vec为向量
    """
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2)


def check_radian(start_radian, end_radian, angle):
    if start_radian >= 0:
        if end_radian >= 0 and end_radian >= start_radian:
            return True if (start_radian <= angle <= end_radian) else False
        elif end_radian >= 0 and end_radian < start_radian:
            return True if not (start_radian <= angle <= end_radian) else False

        elif end_radian <= 0:
            if angle >= 0 and angle >= start_radian:
                return True
            elif angle < 0 and angle <= end_radian:
                return True
            else:
                return False

    elif start_radian < 0:
        if end_radian >= 0:
            if angle >= 0 and angle < end_radian:
                return True
            elif angle < 0 and angle > start_radian:
                return True
            else:
                return False
        elif end_radian < 0 and end_radian > start_radian:
            return (
                True
                if (angle < 0 and start_radian <= angle <= end_radian)
                else False
            )
        elif end_radian < 0 and end_radian < start_radian:
            return True if not (end_radian <= angle <= start_radian) else False

# ===========================================================================
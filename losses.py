import torch


CLASS_NAME = ['cl0', 'cl1', 'cl2','cl3','cl4']

def skew_symmetric(axag_unit):
    """
    Create the skew symmetric matrix for the input vector
    v = (v_1, v_2, v_3)
    v_ss = | 0    -v_3    v_2 |
           | v_3     0   -v_1 |
           | -v_2  v_1     0  |

    :param axag_unit: B, 3 tensor
    :return: B, 3, 3 tensor
    """
    sh = axag_unit.shape
    axag_unit_exp = torch.unsqueeze(torch.unsqueeze(axag_unit, 2), 3)

    row1 = torch.cat([torch.zeros((sh[0], 1, 1), dtype=torch.float64).cuda(),
                      -axag_unit_exp[:, 2, :, :], axag_unit_exp[:, 1, :, :]], dim=2)
    row2 = torch.cat([axag_unit_exp[:, 2, :, :], torch.zeros((sh[0], 1, 1), dtype=torch.float64).cuda(),
                      -axag_unit_exp[:, 0, :, :]], dim=2)

    row3 = torch.cat(
        [-axag_unit_exp[:, 1, :, :], axag_unit_exp[:, 0, :, :], torch.zeros((sh[0], 1, 1), dtype=torch.float64).cuda()],
        dim=2)
    axag_unit_ss = torch.cat([row1, row2, row3], dim=1)
    return axag_unit_ss


def exponential_map(axag, EPS=1e-2):
    """
    Create exponential map for axis-angle representation using Rodrigues' formula
    axag = theta * v_hat
    exp(theta * v_hat) = I + sin(theta)[v_hat]_x + (1 - cos(theta))([v_hat]_x)^2
    For small angle values, use Taylor expansion
    :param axag: B, 3 tensor
    :return: B, 3, 3 tensor
    """
    ss = skew_symmetric(axag)

    theta_sq = torch.sum(axag.pow(2), dim=1)

    is_angle_small = torch.lt(theta_sq, EPS)

    theta = torch.sqrt(theta_sq)

    theta_pow_4 = theta_sq * theta_sq
    theta_pow_6 = theta_sq * theta_sq * theta_sq
    theta_pow_8 = theta_sq * theta_sq * theta_sq * theta_sq

    term_1 = torch.where(is_angle_small,
                         1 - (theta_sq / 6.0) + (theta_pow_4 / 120) - (theta_pow_6 / 5040) + (theta_pow_8 / 362880),
                         torch.sin(theta) / theta)

    term_2 = torch.where(is_angle_small,
                         0.5 - (theta_sq / 24.0) + (theta_pow_4 / 720) - (theta_pow_6 / 40320) + (
                                 theta_pow_8 / 3628800),
                         (1 - torch.cos(theta)) / theta_sq)
    term_1_expand = torch.unsqueeze(torch.unsqueeze(term_1, 1), 2)
    term_2_expand = torch.unsqueeze(torch.unsqueeze(term_2, 1), 2)
    batch_identity = torch.eye(3, dtype=torch.float64).unsqueeze(0).repeat(axag.shape[0], 1, 1).cuda()
    axag_exp = batch_identity + torch.mul(term_1_expand, ss) + torch.mul(term_2_expand, torch.matmul(ss, ss))

    return axag_exp


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    code from https://blog.csdn.net/york1996/article/details/89434935
    """

    result = (t >= t_min).double() * t + (t < t_min).double() * t_min
    result = (result <= t_max).double() * result + (result > t_max).double() * t_max
    return result


def logarithm(R, b_deal_with_sym=False, EPS=1e-2):
    """
    R in SO(3)
    theta = arccos((tr(R)-1)/2)
    ln(R) = (theta/(2*sin(theta)))*(R-R.')
    :param R: B, 3, 3 tensor
    :return: B, 3 tensor
    """
    B = R.shape[0]

    trace_all = torch.zeros(B, dtype=torch.float64).cuda()
    for iii in range(B):
        trace = torch.trace(R[iii, :, :].squeeze(0))
        trace_all[iii] = trace
    trace_temp = (trace_all - 1) / 2

    # take the safe acos
    o_n_e = torch.ones((B,), dtype=torch.float64).cuda()
    trace_temp = clip_by_tensor(trace_temp, -o_n_e, o_n_e)

    theta = torch.acos(trace_temp)

    is_angle_small = torch.lt(theta, EPS)
    theta_pow_2 = theta * theta
    theta_pow_4 = theta_pow_2 * theta_pow_2
    theta_pow_6 = theta_pow_2 * theta_pow_4

    # ss = (R - tf.matrix_transpose(R))
    ss = (R - R.transpose(1, 2))
    mul_expand = torch.where(is_angle_small,
                             0.5 + (theta_pow_2 / 12) + (7 * theta_pow_4 / 720) + (31 * theta_pow_6 / 30240),
                             theta / (2 * torch.sin(theta)))
    if b_deal_with_sym:
        log_R = torch.unsqueeze(torch.unsqueeze(mul_expand, 2), 3) * ss
    else:
        log_R = torch.unsqueeze(torch.unsqueeze(mul_expand, 1), 2) * ss

    return log_R, theta


def get_rotation_error(pred, label):
    '''
    Return (mean) rotation error in form of angular distance in SO(3)
    :param pred: B,3 tensor
    :param label: B,3 tensor
    :return: 1D scalar
    '''
    pred_expMap = exponential_map(pred)

    label_expMap = exponential_map(label)

    R = torch.matmul(label_expMap, pred_expMap.transpose(1, 2))
    R_logMap, loss = logarithm(R)

    return torch.mean(loss), loss


def get_translation_error(pred, label):
    loss_perSample = torch.norm((label - pred), dim=1)
    loss = torch.mean(loss_perSample)
    return loss, loss_perSample


def get_loss(end_points):
    translate_pred = end_points['translate_pred']
    translate_label = end_points['translate_label']
    axag_pred = end_points['axag_pred']
    axag_label = end_points['axag_label']

    point_class = end_points['point_clouds'][:, 0, 3:].double()

    trans_loss, trans_perLoss = get_translation_error(translate_pred.double(), translate_label.double())
    axag_loss, axag_perLoss = get_rotation_error(axag_pred.double(), axag_label.double())
    total_loss = 10 * trans_loss + axag_loss
    total_perloss = 10 * trans_perLoss + axag_perLoss

    trans_perLoss = torch.unsqueeze(trans_perLoss, dim=0).t() * point_class
    trans_clsLoss = torch.sum(trans_perLoss, dim=0)/torch.sum(point_class, dim=0)
    axag_perLoss = torch.unsqueeze(axag_perLoss, dim=0).t() * point_class
    axag_clsLoss = torch.sum(axag_perLoss, dim=0)/torch.sum(point_class, dim=0)
    total_perloss = torch.unsqueeze(total_perloss, dim=0).t() * point_class
    total_clsLoss = torch.sum(total_perloss, dim=0)/torch.sum(point_class, dim=0)

    end_points['trans_loss'] = trans_loss
    end_points['axag_loss'] = axag_loss
    end_points['total_loss'] = total_loss
  
    return total_loss, end_points


if __name__ == '__main__':
    label = torch.tensor([[0.6977, 0.8248, 0.9367]], dtype=torch.float64).cuda()
    pred = torch.tensor([[-2.100418, -2.167796, 0.2733]], dtype=torch.float64).cuda()
    # print(torch.matmul(pred, label))
    print(get_rotation_error(pred, label))
    # import cv2
    # R = cv2.Rodrigues((-2.100418,-2.167796, 0.2733))
    # print(R[0])

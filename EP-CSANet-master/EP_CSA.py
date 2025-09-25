import torch
import torch.nn as nn

import time

import math
import torch.nn.functional as F
from WFT_loss import WFT_loss


def default_conv(in_channels, out_channels, kernel_size, bias=True):
	return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)



class EPM(nn.Module):
	def __init__(self, channel=3, gamma=2, b=0.5):
		super(EPM, self).__init__()

		kernel_size = int(abs((math.log2(channel + 1) / gamma) - b + 1) * 2)

		kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1

		padding = kernel_size // 2
		self.avg = nn.AdaptiveAvgPool2d(1)
		self.conv = nn.Conv1d(
			1, 1, kernel_size=kernel_size, padding=padding, bias=False
		)
		self.sig = nn.Sigmoid()

		self.td = nn.Sequential(
			default_conv(channel, channel, 3),
			default_conv(channel, channel // 1, 3),
			nn.ReLU(inplace=True),
			default_conv(channel // 1, channel, 3),
			nn.Sigmoid()
		)

	def forward(self, x):
		b, c, h, w = x.size()
		y = self.avg(x).view([b, 1, c])
		y = self.conv(y)
		y = self.sig(y).view([b, c, 1, 1])
		a = x * y
		t = self.td(x)
		j = torch.mul((1 - t), a) + torch.mul(t, x)
		return j


class CSPALayer(nn.Module):
	def __init__(self, channel=3):
		super(CSPALayer, self).__init__()
		self.conv1 = nn.Conv2d(channel, channel // 1, 1, padding=0, bias=True)
		self.conv2 = nn.Conv2d(channel, channel // 1, 3, padding=1, bias=True)
		self.conv3 = nn.Sequential(
			nn.Conv2d(channel, channel, 3, padding=1, bias=True),
			nn.ReLU(inplace=True)
		)
		self.conv4 = nn.Conv2d(6, 3, 1, padding=0, bias=True)
		self.pa = nn.Sequential(
			nn.Conv2d(channel, channel // 1, 1, padding=0, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(channel // 1, 3, 1, padding=0, bias=True),
			nn.Sigmoid()
		)
		self.ed = nn.Sequential(
			nn.Conv2d(channel, channel // 1, 1, padding=0, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(channel // 1, 3, 3, padding=1, bias=True)
		)

	def forward(self, x):
		x1 = self.conv1(x)
		x2 = self.pa(x1)
		x3 = self.conv2(x1)
		x4 = x2 * x3
		y1 = self.conv3(x4)
		y2 = self.ed(x)
		concat0 = torch.cat((y1, y2), 1)
		out = self.conv4(concat0)
		x5 = x + out
		return x5

class Block(nn.Module):
	def __init__(self, conv=default_conv, dim=3, kernel_size=1, ):
		super(Block, self).__init__()
		self.conv1 = conv(dim, dim, kernel_size, bias=True)
		self.act1 = nn.ReLU(inplace=True)
		self.conv2 = conv(dim, dim, kernel_size, bias=True)
		self.calayer = CSPALayer(dim)
		self.epm = EPM(dim)

	def forward(self, x):
		res = self.act1(self.conv1(x))
		res = res + x
		res = self.conv2(res)
		res = self.epm(res)
		res = self.calayer(res)
		res += x
		return res


class dehaze_net(nn.Module):

	def __init__(self):
		super(dehaze_net, self).__init__()

		self.relu = nn.ReLU(inplace=True)

		self.e_conv1 = nn.Conv2d(3, 3, 1, 1, 0, bias=True)
		self.e_conv2 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
		self.e_conv3 = nn.Conv2d(6, 3, 5, 1, 2, bias=True)
		self.e_conv4 = nn.Conv2d(6, 3, 7, 1, 3, bias=True)
		self.e_conv5 = nn.Conv2d(12, 3, 3, 1, 1, bias=True)
		self.e_conv6 = Block()



	def forward(self, x):
		source = []
		source.append(x)


		x1 = self.relu(self.e_conv1(x))
		x2 = self.relu(self.e_conv2(x1))
		x3 = self.relu(self.e_conv6(x2))
		concat1 = torch.cat((x2, x3), 1)
		x4 = self.relu(self.e_conv3(concat1))

		concat2 = torch.cat((x3, x4), 1)
		x5 = self.relu(self.e_conv4(concat2))

		concat3 = torch.cat((x2, x3, x4, x5), 1)
		x6 = self.relu(self.e_conv5(concat3))

		clean_image = self.relu((x6 * x) - x6 + 1)

		return clean_image













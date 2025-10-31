import torch
import csv


def compute_mean_std(dataloader):
	channels_sum, channels_squared_sum, num_batches = 0, 0, 0
	for data, _ in dataloader:
		# Mean over batch, height and width, but not over the channels
		channels_sum += torch.mean(data, dim=[0, 2, 3])
		channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
		num_batches += 1
	
	mean = channels_sum / num_batches
	# std = sqrt(E[X^2] - (E[X])^2)
	std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

	with open('./models/mean_std.csv', 'w') as csvfile:
		fieldnames = ['Image', 'Mean', 'Std']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerow({'Image': 'R', 'Mean': mean.tolist()[0], 'Std': std.tolist()[0]})
		writer.writerow({'Image': 'G', 'Mean': mean.tolist()[1], 'Std': std.tolist()[1]})
		writer.writerow({'Image': 'B', 'Mean': mean.tolist()[2], 'Std': std.tolist()[2]})


def read_mean_std(path: str):
	mean = []
	std = []
	with open(path, newline='') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			mean += [float(row['Mean'])]
			std += [float(row['Std'])]

	return mean, std
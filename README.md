# 10707_project
Semantic point cloud segmentation with graph convolutional network

## Usage
First, download data from <https://shapenet.cs.stanford.edu/iccv17>.

Then, install [tensorpack](https://github.com/ppwwyyxx/tensorpack) and [lmdb](https://lmdb.readthedocs.io). Use `write_lmdb.py` to prepare data. For example, `python3 write_lmdb.py train -c Airplane` prepares training data for the Airplane category.

Use `train.py` to train a model and `test.py` to test it. Run `python3 train.py -h` to see available options. Segmentation results can be visualized using `visualize_results.py`.

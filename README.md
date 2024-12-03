# About
This project implements a language model by using contexts and [context mixing](https://en.wikipedia.org/wiki/Context_mixing) to produce an embedding vector.
Each context is a histogram containing the symbol counts found in a circular symbol buffer.
There are eight contexts with circular buffer sizes: 1, 2, 4, 8, 16, 32, 64, and 128 which are fed with 8 bit symbols.
Context mixing is performed with [self attention](https://arxiv.org/abs/1706.03762).
The eight histogram contexts are compressed down to a single embedding vector and then associated with the next symbol.
[Nearest neighbor](https://en.wikipedia.org/wiki/Nearest_neighbor_search) is used for inferring the next symbol for a given embedding.
# Usage
Clone the repo and then:
```sh
go build
```
To build the vector database (1.1GB):
```sh
./txt -build
```
To query the vector database:
```sh
./txt -query "God"
```

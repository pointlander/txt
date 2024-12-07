# About
This project implements a language model by using contexts and [context mixing](https://en.wikipedia.org/wiki/Context_mixing) to produce an embedding vector.
Each context is a histogram containing the symbol counts found in a circular symbol buffer.
There are eight contexts with circular buffer sizes: 1, 2, 4, 8, 16, 32, 64, and 128 which are fed with 8 bit symbols.
Context mixing is performed with [self attention](https://arxiv.org/abs/1706.03762).
The eight histogram contexts are compressed down to a single embedding vector and then associated with the next symbol.
[Nearest neighbor](https://en.wikipedia.org/wiki/Nearest_neighbor_search) is used for inferring the next symbol for a given embedding.
## Mixer
```go
// Mix mixes the histograms
func (m Mixer) Mix() [256]byte {
	mix := [256]byte{}
	x := NewMatrix(256, Size)
	for i := range m.Histograms {
		sum := 0.0
		for _, v := range m.Histograms[i].Vector {
			sum += float64(v)
		}
		for _, v := range m.Histograms[i].Vector {
			x.Data = append(x.Data, float64(v)/sum)
		}
	}
	y := SelfAttention(x, x, x).Sum()
	sum := 0.0
	for _, v := range y.Data {
		sum += v
	}
	for i := range mix {
		mix[i] = byte(128 * y.Data[i] / sum)
	}
	return mix
}
```
# Usage
Clone the repo and then:
```sh
go build
```
To build the vector database (1.1GB):
```sh
./txt -build
```
To query the vector database using nearest neightbor
```sh
./txt -brute -query "God"
```
To query the vector database using approximate nearest neighbor:
```sh
./txt -query "God"
```

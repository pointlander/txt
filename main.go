// Copyright 2024 The TXT Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"embed"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
)

const (
	// Size is the number of histograms
	Size = 8
	// Average is the split average
	Average = 0.12502159179100905
)

//go:embed 10.txt.utf-8.bz2
var Iris embed.FS

// Histogram is a buffered histogram
type Histogram struct {
	Vector [256]byte
	Buffer [128]byte
	Index  int
	Size   int
}

// NewHistogram make a new histogram
func NewHistogram(size int) Histogram {
	h := Histogram{
		Size: size,
	}
	return h
}

// Add adds a symbol to the histogram
func (h *Histogram) Add(s byte) {
	index := (h.Index + 1) % h.Size
	if symbol := h.Buffer[index]; h.Vector[symbol] > 0 {
		h.Vector[symbol]--
	}
	h.Buffer[index] = s
	h.Vector[s]++
	h.Index = index
}

// Mixer mixes several histograms together
type Mixer struct {
	Histograms []Histogram
}

// NewMixer makes a new mixer
func NewMixer() Mixer {
	histograms := make([]Histogram, Size)
	histograms[0] = NewHistogram(1)
	histograms[1] = NewHistogram(2)
	histograms[2] = NewHistogram(4)
	histograms[3] = NewHistogram(8)
	histograms[4] = NewHistogram(16)
	histograms[5] = NewHistogram(32)
	histograms[6] = NewHistogram(64)
	histograms[7] = NewHistogram(128)
	return Mixer{
		Histograms: histograms,
	}
}

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

// Add adds a symbol to a mixer
func (m Mixer) Add(s byte) {
	for i := range m.Histograms {
		m.Histograms[i].Add(s)
	}
}

// TXT is a context
type TXT struct {
	Index  uint64
	Vector [256]byte
	Symbol byte
}

// TXTWriter is a txt file writer
type TXTWriter struct {
	File *os.File
}

// NewTXTWriter creates a new TXTWriter
func NewTXTWriter(file *os.File) TXTWriter {
	return TXTWriter{
		File: file,
	}
}

// Write writes a txt record to the file
func (t *TXTWriter) Write(txt *TXT) {
	_, err := t.File.Write(txt.Vector[:])
	if err != nil {
		panic(err)
	}
	_, err = t.File.Write([]byte{txt.Symbol})
	if err != nil {
		panic(err)
	}
	index := make([]byte, 8)
	for i := 0; i < 8; i++ {
		index[i] = byte(txt.Index >> ((7 - i) * 8))
	}
	_, err = t.File.Write(index)
	if err != nil {
		panic(err)
	}
}

// TXTReader is a txt file reader
type TXTReader struct {
	File *os.File
}

// NewTXTReader make a new TXTReader
func NewTXTReader(file *os.File) TXTReader {
	return TXTReader{
		File: file,
	}
}

// Read reads a txt record
func (t *TXTReader) Read(txt *TXT) bool {
	buffer := make([]byte, 256+1+8)
	n, _ := t.File.Read(buffer)
	if n == 0 {
		return true
	}
	copy(txt.Vector[:], buffer[:256])
	txt.Symbol = buffer[256]
	index := uint64(0)
	for i := 0; i < 8; i++ {
		index <<= 8
		index |= uint64(buffer[257+i])
	}
	txt.Index = index
	return false
}

// Reset resets the reader to the beginning of the file
func (t *TXTReader) Reset() {
	_, err := t.File.Seek(0, 0)
	if err != nil {
		panic(err)
	}
}

// CS is cosine similarity
func (t *TXT) CS(vector *[256]byte) float64 {
	aa, bb, ab := 0.0, 0.0, 0.0
	for i := range vector {
		a, b := float64(vector[i]), float64(t.Vector[i])
		aa += a * a
		bb += b * b
		ab += a * b
	}
	return ab / (math.Sqrt(aa) * math.Sqrt(bb))
}

// CS is float64 cosine similarity
func (t *TXT) CSFloat64(vector *[256]float64) float64 {
	aa, bb, ab := 0.0, 0.0, 0.0
	for i := range vector {
		a, b := vector[i], float64(t.Vector[i])
		aa += a * a
		bb += b * b
		ab += a * b
	}
	return ab / (math.Sqrt(aa) * math.Sqrt(bb))
}

// Splits are the split vectors
var Splits [64][256]float64

func init() {
	rng := rand.New(rand.NewSource(1))
	for i := range Splits {
		for j := range Splits[i] {
			Splits[i][j] = rng.Float64()
		}
	}
}

var (
	// FlagBuild is for building the vector database
	FlagBuild = flag.Bool("build", false, "build the vector database")
	// FlagQuery is for doing a lookup in the database
	FlagQuery = flag.String("query", "In the beginning God created the heaven and the eart", "query for vector database")
	// FlagCount number of symbols to generate
	FlagCount = flag.Int("count", 33, "number of symbols to generate")
)

func main() {
	flag.Parse()

	if *FlagBuild {
		file, err := Iris.Open("10.txt.utf-8.bz2")
		if err != nil {
			panic(err)
		}
		defer file.Close()
		reader := bzip2.NewReader(file)
		data, err := io.ReadAll(reader)
		if err != nil {
			panic(err)
		}
		db, err := os.Create("vectors.bin")
		if err != nil {
			panic(err)
		}
		defer db.Close()
		m := NewMixer()
		avg, count := 0.0, 0.0
		txts := make([]TXT, 0, 8)
		for i, s := range data[:len(data)-1] {
			m.Add(s)
			txt := TXT{}
			txt.Vector = m.Mix()
			txt.Symbol = data[i+1]
			for j := range Splits {
				s := txt.CSFloat64(&Splits[j])
				avg += s
				count++
			}
			txts = append(txts, txt)
		}
		avg /= count
		fmt.Println(avg)
		for i := range txts {
			for j := range Splits {
				txts[i].Index <<= 1
				s := txts[i].CSFloat64(&Splits[j])
				if s > avg {
					txts[i].Index |= 1
				}
			}
		}
		sort.Slice(txts, func(i, j int) bool {
			return txts[i].Index < txts[j].Index
		})

		writer := NewTXTWriter(db)
		for _, txt := range txts {
			writer.Write(&txt)
		}
		return
	}

	input := []byte(*FlagQuery)
	vectors, err := os.Open("vectors.bin")
	if err != nil {
		panic(err)
	}
	defer vectors.Close()
	m := NewMixer()
	for _, s := range input {
		m.Add(s)
	}
	stat, err := vectors.Stat()
	if err != nil {
		panic(err)
	}
	length := stat.Size() / (256 + 1 + 8)
	txt, reader := TXT{}, NewTXTReader(vectors)
	for j := 0; j < *FlagCount; j++ {
		var vector TXT
		vector.Vector = m.Mix()
		for j := range Splits {
			vector.Index <<= 1
			s := vector.CSFloat64(&Splits[j])
			if s > Average {
				vector.Index |= 1
			}
		}
		index := sort.Search(int(length), func(i int) bool {
			_, err := vectors.Seek(int64(i*(256+1+8)), 0)
			if err != nil {
				panic(err)
			}
			txt := TXT{}
			reader.Read(&txt)
			return txt.Index >= vector.Index
		})
		fmt.Println(index)
		reader.Reset()
		//symbol, max := byte(0), -1.0
		_, err := vectors.Seek(int64(index*(256+1+8)), 0)
		if err != nil {
			panic(err)
		}
		reader.Read(&txt)
		symbol := txt.Symbol
		/*for !done {
			s := txt.CS(&vector.Vector)
			if s > max {
				max, symbol = s, txt.Symbol
			}
			done = reader.Read(&txt)
		}*/
		reader.Reset()
		fmt.Printf("%d %s\n", symbol, strconv.Quote(string(symbol)))
		m.Add(symbol)
	}
}

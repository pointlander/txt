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
	"os"
	"strconv"
)

const (
	// Size is the number of histograms
	Size = 8
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
	Vector [256]byte
	Symbol byte
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
		for i, s := range data[:len(data)-1] {
			m.Add(s)
			vector := m.Mix()
			_, err := db.Write(vector[:])
			if err != nil {
				panic(err)
			}
			_, err = db.Write(data[i+1 : i+2])
			if err != nil {
				panic(err)
			}
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
	txt := TXT{}
	sym := make([]byte, 1)
	for j := 0; j < *FlagCount; j++ {
		vector := m.Mix()
		symbol, max := byte(0), -1.0
		for {
			n, _ := vectors.Read(txt.Vector[:])
			if n == 0 {
				break
			}
			if n != 256 {
				panic("didn't get 256 bytes")
			}
			n, _ = vectors.Read(sym)
			if n != 1 {
				panic("missing symbol")
			}
			txt.Symbol = sym[0]
			aa, bb, ab := 0.0, 0.0, 0.0
			for i := range vector {
				a, b := float64(vector[i]), float64(txt.Vector[i])
				aa += a * a
				bb += b * b
				ab += a * b
			}
			s := ab / (math.Sqrt(aa) * math.Sqrt(bb))
			if s > max {
				max, symbol = s, txt.Symbol
			}
		}
		_, err := vectors.Seek(0, 0)
		if err != nil {
			panic(err)
		}
		fmt.Printf("%d %s\n", symbol, strconv.Quote(string(symbol)))
		m.Add(symbol)
	}
}

// Copyright 2024 The TXT Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"embed"
	"fmt"
	"io"
	"math"
)

//go:embed 10.txt.utf-8.bz2
var Iris embed.FS

// Histogram is a buffered histogram
type Histogram struct {
	Vector [256]byte
	Buffer [128]byte
	Index  int
}

// Add adds a symbol to the histogram
func (h *Histogram) Add(s byte) {
	h.Index = (h.Index + 1) % 128
	if symbol := h.Buffer[h.Index]; h.Vector[symbol] > 0 {
		h.Vector[symbol]--
	}
	h.Buffer[h.Index] = s
	h.Vector[s]++
}

// TXT is a context
type TXT struct {
	Vector [256]byte
	Symbol byte
}

func main() {
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
	h := Histogram{}
	txts := make([]TXT, 0, 8)
	for i, s := range data[:len(data)-1] {
		h.Add(s)
		txt := TXT{
			Vector: h.Vector,
			Symbol: data[i+1],
		}
		txts = append(txts, txt)
	}
	input := []byte("In the beginning God created the heaven and the eart")
	h = Histogram{}
	for _, s := range input {
		h.Add(s)
	}
	for j := 0; j < 16; j++ {
		symbol, max := byte(0), -1.0
		for _, txt := range txts {
			aa, bb, ab := 0.0, 0.0, 0.0
			for i := range h.Vector {
				a, b := float64(h.Vector[i]), float64(txt.Vector[i])
				aa += a * a
				bb += b * b
				ab += a * b
			}
			s := ab / (math.Sqrt(aa) * math.Sqrt(bb))
			if s > max {
				max, symbol = s, txt.Symbol
			}
		}
		fmt.Printf("%d %c\n", symbol, symbol)
		h.Add(symbol)
	}
}

// Copyright 2024 The TXT Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"embed"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strconv"
)

const (
	// Size is the number of histograms
	Size = 8
	// Line is the size of a line
	Line = 256 + 2 + 1 + 8
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = 0.01
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

//go:embed 10.txt.utf-8.bz2
var Iris embed.FS

// Markov is a markov model
type Markov [2]byte

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
	Markov     Markov
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
		mix[i] = byte(255 * y.Data[i] / sum)
	}
	return mix
}

// MixFloat64 mixes the histograms outputting float64
func (m Mixer) MixFloat64() [256]float64 {
	mix := [256]float64{}
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
		mix[i] = y.Data[i] / sum
	}
	return mix
}

// Add adds a symbol to a mixer
func (m *Mixer) Add(s byte) {
	for i := range m.Histograms {
		m.Histograms[i].Add(s)
	}
	m.Markov[1] = m.Markov[0]
	m.Markov[0] = s
}

// TXT is a context
type TXT struct {
	Vector [256]byte
	Markov Markov
	Symbol byte
	Index  uint64
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
	_, err = t.File.Write(txt.Markov[:])
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
	buffer := make([]byte, Line)
	n, _ := t.File.Read(buffer)
	if n == 0 {
		return true
	}
	copy(txt.Vector[:], buffer[:256])
	copy(txt.Markov[:], buffer[256:258])
	txt.Symbol = buffer[258]
	index := uint64(0)
	for i := 0; i < 8; i++ {
		index <<= 8
		index |= uint64(buffer[259+i])
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

// CSFloat64 is float64 cosine similarity
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

// CSFloat64 is float64 cosine similarity
func CSFloat64(t *[256]float64, vector *[256]float64) float64 {
	aa, bb, ab := 0.0, 0.0, 0.0
	for i := range vector {
		a, b := vector[i], float64(t[i])
		aa += a * a
		bb += b * b
		ab += a * b
	}
	return ab / (math.Sqrt(aa) * math.Sqrt(bb))
}

// Pow returns the input raised to the current time
func Pow(x float64, i int) float64 {
	y := math.Pow(x, float64(i+1))
	if math.IsNaN(y) || math.IsInf(y, 0) {
		return 0
	}
	return y
}

var (
	// FlagBuild is for building the vector database
	FlagBuild = flag.Bool("build", false, "build the vector database")
	// FlagQuery is for doing a lookup in the database
	FlagQuery = flag.String("query", "In the beginning God created the heaven and the eart", "query for vector database")
	// FlagBrute brute force mode
	FlagBrute = flag.Bool("brute", false, "brute force mode")
	// FlagCount number of symbols to generate
	FlagCount = flag.Int("count", 33, "number of symbols to generate")
)

func float64ToByte(f float64) []byte {
	var buf [8]byte
	binary.BigEndian.PutUint64(buf[:], math.Float64bits(f))
	return buf[:]
}

func byteToFloat64(buf []byte) float64 {
	return math.Float64frombits(binary.BigEndian.Uint64(buf))
}

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
		txts := make([]TXT, 0, 8)
		for i, s := range data[:len(data)-1] {
			m.Add(s)
			txt := TXT{}
			txt.Vector = m.Mix()
			txt.Markov = m.Markov
			txt.Symbol = data[i+1]
			txt.Index = uint64(i)
			txts = append(txts, txt)
		}

		sort.Slice(txts, func(i, j int) bool {
			if txts[i].Markov[0] < txts[j].Markov[0] {
				return true
			} else if txts[i].Markov[0] == txts[j].Markov[0] {
				return txts[i].Markov[1] < txts[j].Markov[1]
			}
			return false
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
	if *FlagBrute {
		txt, reader := TXT{}, NewTXTReader(vectors)
		for j := 0; j < *FlagCount; j++ {
			vector := m.MixFloat64()
			symbol, max := byte(0), -1.0
			done := reader.Read(&txt)
			for !done {
				s := txt.CSFloat64(&vector)
				if s > max {
					max, symbol = s, txt.Symbol
				}
				done = reader.Read(&txt)
			}
			reader.Reset()
			fmt.Printf("%d %s\n", symbol, strconv.Quote(string(symbol)))
			m.Add(symbol)
		}
		return
	}
	stat, err := vectors.Stat()
	if err != nil {
		panic(err)
	}
	length := stat.Size() / Line
	txt, reader := TXT{}, NewTXTReader(vectors)
	for j := 0; j < *FlagCount; j++ {
		index := sort.Search(int(length), func(i int) bool {
			_, err := vectors.Seek(int64(i*Line), 0)
			if err != nil {
				panic(err)
			}
			txt := TXT{}
			reader.Read(&txt)
			if txt.Markov[0] > m.Markov[0] {
				return true
			} else if txt.Markov[0] == m.Markov[0] {
				return txt.Markov[1] >= m.Markov[1]
			}
			return false
		})
		symbol, max := byte(0), -1.0
		vector := m.MixFloat64()
		for k := 0; k < 2048; k++ {
			index := index + k
			if index < 0 {
				continue
			} else if index >= int(length) {
				continue
			}
			_, err := vectors.Seek(int64(index*Line), 0)
			if err != nil {
				panic(err)
			}
			reader.Read(&txt)
			s := txt.CSFloat64(&vector)
			if s > max {
				max, symbol = s, txt.Symbol
			}
		}
		fmt.Printf("%d %s\n", symbol, strconv.Quote(string(symbol)))
		m.Add(symbol)
	}
}

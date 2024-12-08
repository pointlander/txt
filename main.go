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
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/pointlander/gradient/tf64"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

const (
	// Size is the number of histograms
	Size = 8
	// Line is the size of a line
	Line = 256 + 1 + 8
	// Average is the split average
	Average = 0.32957396832954744
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
	buffer := make([]byte, Line)
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

// GenerateSplits generates the splitting vectors
func GenerateSplits(txts []TXT) (splits [64][256]float64) {
	rng := rand.New(rand.NewSource(1))
	for s := range splits {
		points := make(plotter.XYs, 0, 8)
		set := tf64.NewSet()
		set.Add("w1", 256, 1)

		for i := range set.Weights {
			w := set.Weights[i]
			if strings.HasPrefix(w.N, "b") {
				w.X = w.X[:cap(w.X)]
				w.States = make([][]float64, StateTotal)
				for i := range w.States {
					w.States[i] = make([]float64, len(w.X))
				}
				continue
			}
			factor := math.Sqrt(float64(w.S[0]))
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, rng.NormFloat64()*factor)
			}
			w.States = make([][]float64, StateTotal)
			for i := range w.States {
				w.States[i] = make([]float64, len(w.X))
			}
		}

		others := tf64.NewSet()
		others.Add("input", 256, 1)
		others.Add("output", 1, 1)

		for i := range others.Weights {
			w := others.Weights[i]
			w.X = w.X[:cap(w.X)]
		}
		others.ByName["output"].X[0] = .33

		l1 := tf64.Similarity(tf64.Abs(set.Get("w1")), others.Get("input"))
		loss := tf64.Quadratic(l1, others.Get("output"))

		for i := range txts {
			others.Zero()
			set.Zero()
			input := others.ByName["input"].X
			for j := range input {
				input[j] = float64(txts[i].Vector[j])
			}
			cost := tf64.Gradient(loss).X[0]

			norm := 0.0
			for _, p := range set.Weights {
				for _, d := range p.D {
					norm += d * d
				}
			}
			norm = math.Sqrt(norm)
			b1, b2 := Pow(B1, i), Pow(B2, i)
			scaling := 1.0
			if norm > 1 {
				scaling = 1 / norm
			}
			for _, w := range set.Weights {
				for l, d := range w.D {
					g := d * scaling
					m := B1*w.States[StateM][l] + (1-B1)*g
					v := B2*w.States[StateV][l] + (1-B2)*g*g
					w.States[StateM][l] = m
					w.States[StateV][l] = v
					mhat := m / (1 - b1)
					vhat := v / (1 - b2)
					if vhat < 0 {
						vhat = 0
					}
					w.X[l] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
				}
			}
			points = append(points, plotter.XY{X: float64(i), Y: float64(cost)})
		}
		fmt.Println(s)

		w1 := set.ByName["w1"].X
		for i := range w1 {
			splits[s][i] = w1[i]
		}

		p := plot.New()

		p.Title.Text = "epochs vs cost"
		p.X.Label.Text = "epochs"
		p.Y.Label.Text = "cost"

		scatter, err := plotter.NewScatter(points)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		p.Add(scatter)

		err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("epochs%d.png", s))
		if err != nil {
			panic(err)
		}
	}
	return splits
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
		avg, count := 0.0, 0.0
		txts := make([]TXT, 0, 8)
		for i, s := range data[:len(data)-1] {
			m.Add(s)
			txt := TXT{}
			txt.Vector = m.Mix()
			txt.Symbol = data[i+1]
			txts = append(txts, txt)
		}

		splits := GenerateSplits(txts)
		for i := range splits {
			for j := range splits[i] {
				splits[i][j] = math.Abs(splits[i][j])
			}
		}
		splitsFile, err := os.Create("splits.bin")
		if err != nil {
			panic(err)
		}
		defer splitsFile.Close()
		for i := range splits {
			for j := range splits[i] {
				_, err := splitsFile.Write(float64ToByte(splits[i][j]))
				if err != nil {
					panic(err)
				}
			}
		}

		for i := range txts {
			for j := range splits {
				s := txts[i].CSFloat64(&splits[j])
				avg += s
				count++
			}
		}
		avg /= count
		fmt.Println(avg)

		for i := range txts {
			for j := range splits {
				txts[i].Index <<= 1
				s := txts[i].CSFloat64(&splits[j])
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
	splitsFile, err := os.Open("splits.bin")
	if err != nil {
		panic(err)
	}
	defer splitsFile.Close()
	var splits [64][256]float64
	buffer := make([]byte, 8)
	for i := range splits {
		for j := range splits[i] {
			n, _ := splitsFile.Read(buffer)
			if n != 8 {
				panic("there should be 8 bytes")
			}
			splits[i][j] = byteToFloat64(buffer)
		}
	}

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
		vector, index := m.MixFloat64(), uint64(0)
		for j := range splits {
			index <<= 1
			s := CSFloat64(&vector, &splits[j])
			if s > Average {
				index |= 1
			}
		}
		symbol, max := byte(0), -1.0
		search := func(query uint64) {
			index := sort.Search(int(length), func(i int) bool {
				_, err := vectors.Seek(int64(i*Line), 0)
				if err != nil {
					panic(err)
				}
				txt := TXT{}
				reader.Read(&txt)
				return txt.Index >= query
			})
			reader.Reset()

			_, err := vectors.Seek(int64(index*Line), 0)
			if err != nil {
				panic(err)
			}
			done := reader.Read(&txt)
			last := txt.Index
			for !done {
				s := txt.CSFloat64(&vector)
				if s > max {
					max, symbol = s, txt.Symbol
				}
				done = reader.Read(&txt)
				if last != txt.Index {
					break
				}
				last = txt.Index
			}
			reader.Reset()
		}
		search(index)
		for k := 0; k < 64; k++ {
			query := index ^ (1 << k)
			search(query)
		}
		fmt.Printf("%d %s\n", symbol, strconv.Quote(string(symbol)))
		m.Add(symbol)
	}
}

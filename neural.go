// Copyright 2024 The TXT Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/pointlander/gradient/tf64"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

// Neural is a neural network
type Neural struct {
	Set    tf64.Set
	Others tf64.Set
	L1     tf64.Meta
	L2     tf64.Meta
	L3     tf64.Meta
	L4     tf64.Meta
	//L5     tf64.Meta
	//L6     tf64.Meta
	Loss tf64.Meta
}

// Load loads a neural network from a file
func Load() Neural {
	set := tf64.NewSet()
	cost, epochs, err := set.Open("set.db")
	if err != nil {
		panic(err)
	}
	fmt.Println(cost, epochs)

	others := tf64.NewSet()
	others.Add("input", 256)
	others.Add("output", 256)

	for i := range others.Weights {
		w := others.Weights[i]
		w.X = w.X[:cap(w.X)]
	}

	l1 := tf64.Everett(tf64.Add(tf64.Mul(set.Get("w1"), others.Get("input")), set.Get("b1")))
	l2 := tf64.Everett(tf64.Add(tf64.Mul(set.Get("w2"), l1), set.Get("b2")))
	l3 := tf64.Everett(tf64.Add(tf64.Mul(set.Get("w3"), l2), set.Get("b3")))
	l4 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w4"), l3), set.Get("b4")))
	loss := tf64.Quadratic(l4, others.Get("output"))

	/*sumRows := tf64.U(SumRows)

	query := tf64.Mul(set.Get("query"), others.Get("input"))
	key := tf64.Mul(set.Get("key"), others.Get("input"))
	value := tf64.Mul(set.Get("value"), others.Get("input"))
	l1 := tf64.Add(others.Get("input"), tf64.T(tf64.Mul(tf64.Softmax(tf64.Mul(query, key)), tf64.T(value))))
	l2 := tf64.Everett(tf64.Add(tf64.Mul(set.Get("w1"), l1), set.Get("b1")))
	l3 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w2"), l2), set.Get("b2")))
	query2 := tf64.Mul(set.Get("query2"), l3)
	key2 := tf64.Mul(set.Get("key2"), l3)
	value2 := tf64.Mul(set.Get("value2"), l3)
	l4 := tf64.Add(l3, tf64.T(tf64.Mul(tf64.Softmax(tf64.Mul(query2, key2)), tf64.T(value2))))
	l5 := tf64.Everett(tf64.Add(tf64.Mul(set.Get("w3"), l4), set.Get("b3")))
	l6 := tf64.Sigmoid(sumRows(tf64.Add(tf64.Mul(set.Get("w4"), l5), set.Get("b4"))))
	loss := tf64.Quadratic(l6, others.Get("output"))*/

	return Neural{
		Set:    set,
		Others: others,
		L1:     l1,
		L2:     l2,
		L3:     l3,
		L4:     l4,
		//L5:     l5,
		//L6:     l6,
		Loss: loss,
	}
}

// SumRows sums the rows of the matrix
func SumRows(k tf64.Continuation, node int, a *tf64.V, options ...map[string]interface{}) bool {
	size, width := len(a.X), a.S[0]
	total := 0.0
	c := tf64.NewV(width)
	c.X = c.X[:cap(c.X)]
	for i := 0; i < size; i += width {
		for j, ax := range a.X[i : i+width] {
			c.X[j] += ax
		}
		total++
	}
	for i := range c.X {
		c.X[i] /= total
	}
	if k(&c) {
		return true
	}
	for i := 0; i < size; i += width {
		for j := range a.D[i : i+width] {
			a.D[i+j] += c.D[j] / total
		}
	}
	return false
}

// Learn learn a neural network
func Learn(data []byte) Neural {
	rng := rand.New(rand.NewSource(1))
	set := tf64.NewSet()
	/*set.Add("query", 256, 256)
	set.Add("key", 256, 256)
	set.Add("value", 256, 256)
	set.Add("query2", 256, 256)
	set.Add("key2", 256, 256)
	set.Add("value2", 256, 256)
	set.Add("w1", 256, 256)
	set.Add("b1", 256)
	set.Add("w2", 512, 256)
	set.Add("b2", 256)
	set.Add("w3", 256, 256)
	set.Add("b3", 256)
	set.Add("w4", 512, 256)
	set.Add("b4", 256)*/
	set.Add("w1", 256, 256)
	set.Add("b1", 256)
	set.Add("w2", 512, 256)
	set.Add("b2", 256)
	set.Add("w3", 512, 256)
	set.Add("b3", 256)
	set.Add("w4", 512, 256)
	set.Add("b4", 256)

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
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, rng.NormFloat64()*factor)
		}
		w.States = make([][]float64, StateTotal)
		for i := range w.States {
			w.States[i] = make([]float64, len(w.X))
		}
	}

	others := tf64.NewSet()
	//others.Add("input", 256, Size)
	others.Add("input", 256)
	others.Add("output", 256)

	for i := range others.Weights {
		w := others.Weights[i]
		w.X = w.X[:cap(w.X)]
	}

	l1 := tf64.Everett(tf64.Add(tf64.Mul(set.Get("w1"), others.Get("input")), set.Get("b1")))
	l2 := tf64.Everett(tf64.Add(tf64.Mul(set.Get("w2"), l1), set.Get("b2")))
	l3 := tf64.Everett(tf64.Add(tf64.Mul(set.Get("w3"), l2), set.Get("b3")))
	l4 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w4"), l3), set.Get("b4")))
	loss := tf64.Quadratic(l4, others.Get("output"))

	/*sumRows := tf64.U(SumRows)

	options := map[string]interface{}{
		"rng": rng,
	}
	query := tf64.Mul(set.Get("query"), others.Get("input"))
	key := tf64.Mul(set.Get("key"), others.Get("input"))
	value := tf64.Mul(set.Get("value"), others.Get("input"))
	l1 := tf64.Add(others.Get("input"), tf64.T(tf64.Mul(tf64.Softmax(tf64.Mul(query, key)), tf64.T(value))))
	l2 := tf64.Everett(tf64.Add(tf64.Mul(set.Get("w1"), tf64.Dropout(l1, options)), set.Get("b1")))
	l3 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w2"), tf64.Dropout(l2, options)), set.Get("b2")))
	query2 := tf64.Mul(set.Get("query2"), l3)
	key2 := tf64.Mul(set.Get("key2"), l3)
	value2 := tf64.Mul(set.Get("value2"), l3)
	l4 := tf64.Add(l3, tf64.T(tf64.Mul(tf64.Softmax(tf64.Mul(query2, key2)), tf64.T(value2))))
	l5 := tf64.Everett(tf64.Add(tf64.Mul(set.Get("w3"), tf64.Dropout(l4, options)), set.Get("b3")))
	l6 := tf64.Sigmoid(sumRows(tf64.Add(tf64.Mul(set.Get("w4"), tf64.Dropout(l5, options)), set.Get("b4"))))
	loss := tf64.Quadratic(l6, others.Get("output"))*/

	last, epochs := 0.0, 0
	points := make(plotter.XYs, 0, 8)
	fmt.Println("learning:", len(data))
	for i := 0; i < len(data); i++ {
		pow := func(x float64) float64 {
			y := math.Pow(x, float64(i+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return y
		}

		others.Zero()
		index := rng.Intn(len(data) - 256)
		m := NewMixer()
		end := index + 8 + rng.Intn(120)
		for j := index; j < end; j++ {
			m.Add(data[j])
		}
		vector := m.MixFloat64() //m.Raw()
		input := others.ByName["input"].X
		for j := range input {
			input[j] = vector[j]
		}
		output := others.ByName["output"].X
		for j := range output {
			output[j] = .001
		}
		output[data[end]] = 1

		set.Zero()
		cost := tf64.Gradient(loss).X[0]
		last, epochs = cost, i
		if math.IsNaN(cost) || math.IsInf(cost, 0) {
			break
		}

		norm := 0.0
		for _, p := range set.Weights {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = math.Sqrt(norm)
		b1, b2 := pow(B1), pow(B2)
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
		if i%1024 == 0 {
			fmt.Println(i, cost)
		}
	}

	err := set.Save("set.db", last, epochs)
	if err != nil {
		panic(err)
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

	err = p.Save(8*vg.Inch, 8*vg.Inch, "epochs.png")
	if err != nil {
		panic(err)
	}
	return Neural{
		Set:    set,
		Others: others,
		L1:     l1,
		L2:     l2,
		L3:     l3,
		L4:     l4,
		//L5:     l5,
		//L6:     l6,
		Loss: loss,
	}
}

// Inference performs inference of the neural network
func (n *Neural) Inference(input [256]float64) int {
	symbol, max := 0, 0.0
	in := n.Others.ByName["input"].X
	for i := range in {
		in[i] = input[i]
	}
	n.L4(func(a *tf64.V) bool {
		for i, v := range a.X {
			if v > max {
				max, symbol = v, i
			}
		}
		return true
	})

	return symbol
}

// Distribution performs inference of the neural network
func (n *Neural) Distribution(input []float64) (d []float64) {
	in := n.Others.ByName["input"].X
	for i := range in {
		in[i] = input[i]
	}
	n.L4(func(a *tf64.V) bool {
		d = a.X
		return true
	})

	return d
}

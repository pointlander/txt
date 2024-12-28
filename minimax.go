// Copyright 2024 The TXT Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
)

// Depth is the minimax search depth
const Depth = 2

// Softmax computes the softmax of a vector
func Softmax(vector []float64, T float64) {
	max := 0.0
	for _, v := range vector {
		v /= T
		if v > max {
			max = v
		}
	}
	s := max * S
	sum := 0.0
	values := make([]float64, len(vector))
	for j, value := range vector {
		values[j] = math.Exp(value/T - s)
		sum += values[j]
	}
	for j, value := range values {
		vector[j] = value / sum
	}
}

// Max is the minimax max function
func Max(neural *Neural, depth int, action byte, m *Mixer) float64 {
	if depth >= Depth {
		cp := m.Copy()
		cp.Add(action)
		vector := cp.MixFloat64()
		histogram := neural.Distribution(vector)
		Softmax(histogram, 1)
		e := 0.0
		for _, v := range histogram {
			if v == 0 {
				continue
			}
			e += v * math.Log(v)
		}
		return -e
		/*avg, count := 0.0, 0.0
		for i := range histogram {
			avg += float64(histogram[i]) / sum
			count++
		}
		avg /= count
		stddev := 0.0
		for i := range histogram {
			diff := (float64(histogram[i]) / sum) - avg
			stddev += diff * diff
		}
		stddev = math.Sqrt(stddev / count)
		return stddev*/
	}
	cp := m.Copy()
	cp.Add(action)
	vector := cp.MixFloat64()
	histogram := neural.Distribution(vector)
	Softmax(histogram, 1)
	max := 0.0
	for i, v := range histogram {
		if v > 1.0/256.0 {
			x := Min(neural, depth+1, byte(i), &cp)
			if x > max {
				max = x
			}
		}
	}
	return max
}

// Min is the minimax min function
func Min(neural *Neural, depth int, action byte, m *Mixer) float64 {
	if depth >= Depth {
		cp := m.Copy()
		cp.Add(action)
		vector := cp.MixFloat64()
		histogram := neural.Distribution(vector)
		Softmax(histogram, 1)
		e := 0.0
		for _, v := range histogram {
			if v == 0 {
				continue
			}
			e += v * math.Log(v)
		}
		return -e
		/*avg, count := 0.0, 0.0
		for i := range histogram {
			avg += float64(histogram[i]) / sum
			count++
		}
		avg /= count
		stddev := 0.0
		for i := range histogram {
			diff := (float64(histogram[i]) / sum) - avg
			stddev += diff * diff
		}
		stddev = math.Sqrt(stddev / count)
		return stddev*/
	}
	cp := m.Copy()
	cp.Add(action)
	vector := cp.MixFloat64()
	histogram := neural.Distribution(vector)
	Softmax(histogram, 1)
	min := math.MaxFloat64
	for i, v := range histogram {
		if v > 1.0/256.0 {
			x := Max(neural, depth+1, byte(i), &cp)
			if x < min {
				min = x
			}
		}
	}
	return min
}

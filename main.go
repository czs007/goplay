package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"fmt"
	milvusClient "github.com/xiaocai2333/milvus-sdk-go/v2/client"
	"github.com/xiaocai2333/milvus-sdk-go/v2/entity"
	"io"
	"math"
	"os"
	"sync"
	"time"
)

const (
	CollectionName = "taip"
	PartitionName = "_default"
	Dim = 768
	QueryFile = "query.npy"
	DataPath = "/data/milvus/raw_data/zjlab"
	RunTime = 1000
	Goroutine = 1
)

var (
	TopK = []int{50}
	NQ = []int{1}
	EF = []int{50}
	allQPS = 0.0
)


func createClient() milvusClient.Client{
	client, err := milvusClient.NewGrpcClient(context.Background(), "172.18.50.4:19530")
	if err != nil {
		panic(err)
		//fmt.Println("create milvus client failed, err = ", err)
		//return nil
	}
	//defer func(client milvusClient.Client) {
	//	err := client.Close()
	//	if err != nil {
	//		//fmt.Println("close milvus client failed, err = ", err)
	//		panic(err)
	//	}
	//}(client)
	return client
}

func BytesToFloat32(bits []byte) []float32 {
	vectors := make([]float32, 0)
	start, end := 0, 4
	for start < len(bits) && end <= len(bits) {
		num := math.Float32frombits(binary.LittleEndian.Uint32(bits[start:end]))
		vectors = append(vectors, num)
		start += 4
		end += 4
	}
	return vectors
}

func ReadBytesFromFile(nq int) []byte {
	filePath := DataPath+"/"+QueryFile
	f, err := os.Open(filePath)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	r := bufio.NewReader(f)
	chunks := make([]byte, 0)
	buf := make([]byte, 4*Dim)
	readByte := 0
	for readByte < nq*Dim*4 {
		n, err := r.Read(buf)
		if err != nil && err != io.EOF {
			panic(err)
		}
		if 0 == n {
			break
		}
		chunks = append(chunks, buf[:n]...)
		readByte += n
	}
	//fmt.Println(len(chunks))
	return chunks[:nq*Dim*4]
}

func generatedEntities(nq int) []entity.Vector {
	bits := ReadBytesFromFile(nq)
	vectors := make([]entity.Vector, 0)
	for i := 0; i < nq; i++ {
		var vector entity.FloatVector = BytesToFloat32(bits[i*Dim*4:(i+1)*Dim*4])
		//fmt.Println(len(vector))
		vectors = append(vectors, vector)
	}
	return vectors
}

//func search(client milvusClient.Client) {
//	has, err := client.HasCollection(context.Background(), "taip")
//	if err != nil {
//		fmt.Println("Get collection failed, err = ", err)
//		return
//	}
//	if !has {
//		fmt.Println("Get collection failed, collection is not exist")
//		return
//	}
//
//	for _, nq := range NQ {
//		vectors := generatedEntities(nq)
//		for _, ef := range EF {
//			searchParams, err := entity.NewIndexHNSWSearchParam(ef)
//			if err != nil {
//				panic(err)
//			}
//			for _, topK := range TopK {
//				allQPS := 0.0
//				var wg sync.WaitGroup
//				for g := 0; g < Goroutine; g++ {
//					wg.Add(1)
//					go func() {
//						defer wg.Done()
//						cost := int64(0)
//						for i := 0; i < RunTime; i++ {
//							start := time.Now()
//							_, err := client.Search(context.Background(), CollectionName, []string{PartitionName}, "", []string{},
//								vectors, "vec", entity.L2, topK, searchParams)
//							if err != nil {
//								panic(err)
//							}
//							cost += time.Since(start).Microseconds()
//						}
//						avgTime := float64(cost/RunTime)/1000.0/1000.0
//						qps := float64(nq)/avgTime
//						fmt.Printf("average search time: %f， vps: %f \n", avgTime, qps)
//						allQPS += qps
//					}()
//				}
//				wg.Wait()
//				fmt.Printf("nq = %d, topK = %d, ef = %d, goroutine = %d, vps = %f \n", nq, topK, ef, Goroutine, allQPS)
//			}
//		}
//	}
//}


func search() {
	for _, nq := range NQ {
		vectors := generatedEntities(nq)
		for _, ef := range EF {
			searchParams, err := entity.NewIndexHNSWSearchParam(ef)
			if err != nil {
				panic(err)
			}
			for _, topK := range TopK {
				var wg sync.WaitGroup
				wg.Add(Goroutine)
				for g := 0; g < Goroutine; g++ {
					go func() {
						defer wg.Done()
						client := createClient()
						defer client.Close()
						has, err := client.HasCollection(context.Background(), "taip")
						if err != nil {
							fmt.Println("Get collection failed, err = ", err)
							return
						}
						if !has {
							fmt.Println("Get collection failed, collection is not exist")
							return
						}
						cost := int64(0)
						for i := 0; i < RunTime; i++ {
							start := time.Now()
							_, err := client.Search(context.Background(), CollectionName, []string{PartitionName}, "", []string{},
								vectors, "vec", entity.L2, topK, searchParams)
							if err != nil {
								panic(err)
							}
							searchCost := time.Since(start).Microseconds()
							fmt.Printf("search cost: %d \n", searchCost)
							cost += searchCost
						}
						avgTime := float64(cost/RunTime)/1000.0/1000.0
						qps := float64(nq)/avgTime
						fmt.Printf("average search time: %f， vps: %f \n", avgTime, qps)
						allQPS += qps
					}()
				}
				wg.Wait()
				fmt.Printf("nq = %d, topK = %d, ef = %d, goroutine = %d, vps = %f \n", nq, topK, ef, Goroutine, allQPS)
			}
		}
	}
}

func main() {
	search()
	//defer client.Close()
	//var wg sync.WaitGroup
	//for i := 0; i < Goroutine; i++ {
	//	wg.Add(1)
	//	go func() {
	//		defer wg.Done()
	//		search()
	//	}()
	//}
	//wg.Wait()
	//client := createClient()
	//
	//search(client)
	//client.Close()
}

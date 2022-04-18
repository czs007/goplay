package main

import (
	"context"
	"fmt"
	milvusClient "github.com/xiaocai2333/milvus-sdk-go/v2/client"
	"github.com/xiaocai2333/milvus-sdk-go/v2/entity"
	"sync"
	"time"
)

var (
	TopK = []int{50}
	NQ = []int{1}
	EF = []int{50}
	NPROBE = []int{1, 10, 1024}
	allQPS = 0.0
)


func Search(client milvusClient.Client, dataset, indexType string, process int) {
	var pList []int
	if indexType == "HNSW" {
		pList = EF
	}else if indexType == "IVF_FLAT" {
		pList = NPROBE
	}else {
		panic("illegal index type")
	}

	var dataPath string
	if dataset == "taip" {
		dataPath = TaipDataPath
	}else if dataset == "sift" {
		dataPath = SiftDataPath
	}else {
		panic("wrong dataset")
	}
	for _, p := range pList {
		searchParams := newSearchParams(p, indexType)
		for _, nq := range NQ {
			vectors := generatedEntities(dataPath, nq)
			for _, topK := range TopK {
				allQPS = 0.0
				var wg sync.WaitGroup
				wg.Add(process)
				for g := 0; g < process; g++ {
					go func() {
						defer wg.Done()
						cost := int64(0)
						for i := 0; i < RunTime; i++ {
							start := time.Now()
							_, err := client.Search(context.Background(), dataset, []string{}, "", []string{},
								vectors, VecFieldName, entity.L2, topK, searchParams)
							if err != nil {
								panic(err)
							}
							cost += time.Since(start).Microseconds()
						}
						avgTime := float64(cost/RunTime)/1000.0/1000.0
						qps := float64(nq)/avgTime
						fmt.Printf("average search time: %f， vps: %f \n", avgTime, qps)
						allQPS += qps
					}()
				}
				wg.Wait()
				fmt.Printf("nq = %d, topK = %d, param = %d, goroutine = %d, vps = %f \n", nq, topK, p, process, allQPS)
			}
		}
	}
}


func newSearchParams(p int, indexType string) entity.SearchParam {
	if indexType == "HNSW" {
		searchParams, err := entity.NewIndexHNSWSearchParam(p)
		if err != nil {
			panic(err)
		}
		return searchParams
	}else if indexType == "IVF_FLAT" {
		searchParams, err := entity.NewIndexIvfFlatSearchParam(p)
		if err != nil {
			panic(err)
		}
		return searchParams
	}
	panic("illegal search params")
}

//func search() {
//	for _, nq := range NQ {
//		vectors := generatedEntities(nq)
//		for _, ef := range EF {
//			searchParams, err := entity.NewIndexHNSWSearchParam(ef)
//			if err != nil {
//				panic(err)
//			}
//			for _, topK := range TopK {
//				var wg sync.WaitGroup
//				wg.Add(Goroutine)
//				for g := 0; g < Goroutine; g++ {
//					go func() {
//						defer wg.Done()
//						client := createClient()
//						defer client.Close()
//						//has, err := client.HasCollection(context.Background(), "taip")
//						//if err != nil {
//						//	fmt.Println("Get collection failed, err = ", err)
//						//	return
//						//}
//						//if !has {
//						//	fmt.Println("Get collection failed, collection is not exist")
//						//	return
//						//}
//						//for i := 0; i < 5; i++ {
//						//	_, err := client.Search(context.Background(), CollectionName, []string{}, "", []string{},
//						//		vectors, "vec", entity.L2, topK, searchParams)
//						//	if err != nil {
//						//		panic(err)
//						//	}
//						//}
//						//time.Sleep(5*time.Second)
//						cost := int64(0)
//						for i := 0; i < RunTime; i++ {
//							start := time.Now().UnixMicro()
//							//fmt.Printf("search start1 time: %d  \n", start)
//
//							_, err := client.Search(context.Background(), CollectionName, []string{DefaultPartitionName}, "", []string{},
//								vectors, "vec", entity.L2, topK, searchParams)
//							if err != nil {
//								panic(err)
//							}
//							end := time.Now().UnixMicro()
//							fmt.Printf("search start time: %d,  search end time: %d  search cost: %d \n", start, end, end-start)
//							cost += end-start
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

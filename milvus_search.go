package main

import (
	"context"
	"fmt"
	"sync"
	"time"

	milvusClient "github.com/zhagnlu/milvus-sdk-go/v2/client"
	"github.com/zhagnlu/milvus-sdk-go/v2/entity"
)

var (
	TopK   = []int{1}
	NQ     = []int{1}
	EF     = []int{64}
	NPROBE = []int{16}
)

func Search(client milvusClient.Client, dataset, indexType string, process int, partitions []string) {
	var pList []int
	if indexType == "HNSW" {
		pList = EF
	} else if indexType == "IVF_FLAT" {
		pList = NPROBE
	}

	var dataPath string
	if dataset == "taip" || dataset == "zc" {
		dataPath = TaipDataPath
	} else if dataset == "sift" {
		dataPath = SiftDataPath
	} else {
		panic("wrong dataset")
	}
	for _, p := range pList {
		searchParams := newSearchParams(p, indexType)
		for _, nq := range NQ {
			vectors := generatedEntities(dataPath, nq)
			for _, topK := range TopK {
				var wg sync.WaitGroup

				wg.Add(process)
				start := time.Now()
				for g := 0; g < process; g++ {
					go func() {
						defer wg.Done()
						for i := 0; i < RunCount; i++ {
							_, err := client.Search(context.Background(), dataset, partitions, "", []string{},
								vectors, VecFieldName, entity.L2, topK, searchParams, 1)
							if err != nil {
								panic(err)
							}
							// fmt.Printf("Result len:%d\n", len(result))
							// for _, res := range result {
							// 	fmt.Printf("resultCount%d\n", res.ResultCount)
							// 	fmt.Printf("IDs len %d\n", res.IDs.Len())
							// 	fmt.Printf("scoresCount%d\n", len(res.Scores))
							// 	for index, score := range res.Scores {
							// 		fmt.Printf("scores %d:%f\n", index, score)

							// 	}
							// }
						}
					}()
				}

				wg.Wait()
				duration := time.Since(start).Microseconds()
				avgTime := float64(duration/RunCount) / 1000.0 / 1000.0
				allReq := RunCount * process
				allQPS := float64(allReq) / (float64(duration) / 1000 / 1000)
				fmt.Printf("nq = %d, topK = %d, param = %d, goroutine = %d, avgtime = %f, allNq:%d, qps = %f \n", nq, topK, p, process, avgTime, allReq, allQPS)
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
	} else if indexType == "IVF_FLAT" {
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

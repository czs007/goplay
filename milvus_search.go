package main

import (
	"context"
	"fmt"
	milvusClient "github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"sync"
	"time"
)

var (
	TopK = []int{10}
	NQ = []int{1}
	EF = []int{23}
	NPROBE = []int{1, 10, 1024}
	allQPS = 0.0
)


func Search(client milvusClient.Client, dataset, indexType string, process int, partitions []string) {
	var pList []int
	indexType = "HNSW"
	pList = EF

	var dataPath string
	dataPath = SiftDataPath
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
							ret, err := client.Search(context.Background(), dataset, partitions, "", []string{},
								vectors, VecFieldName, entity.L2, topK, searchParams,milvusClient.WithSearchQueryConsistencyLevel(entity.ClEventually))
							if err != nil {
								panic(err)
							//fmt.Println(err)
							}
							if len(ret) == 0 {
								//fmt.Println("empty result")
							} else {

								fmt.Println(" not empty result")

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

package main

import (
	"context"
	"fmt"
	"time"

	milvusClient "github.com/zhagnlu/milvus-sdk-go/v2/client"
	"github.com/zhagnlu/milvus-sdk-go/v2/entity"
)

var (
	CurIndexType   = "HNSW"
	CurIndexRows   = int64(0)
	TotalIndexRows = int64(0)
)

func CreateIndex(client milvusClient.Client, dataset string, indexType string) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if entity.IndexType(indexType) == entity.Flat {
		if err := client.DropIndex(ctx, dataset, VecFieldName); err != nil {
			panic(err)
		}
		return
	}
	CurIndexType = indexType
	_ = client.Flush(ctx, dataset, false)
	fmt.Printf("flush:%s done \n", dataset)
	if dataset == "taip" || dataset == "zc" {
		if entity.IndexType(indexType) == entity.HNSW {
			if err := client.CreateIndex(ctx, dataset, VecFieldName, NewTaipHNSWIndex(), true); err != nil {
				panic(err)
			}
		} else if entity.IndexType(indexType) == entity.IvfFlat {
			if err := client.CreateIndex(ctx, dataset, VecFieldName, NewTaipIVFFLATIndex(), true); err != nil {
				panic(err)
			}
		}
	} else if dataset == "sift" {
		if entity.IndexType(indexType) == entity.HNSW {
			if err := client.CreateIndex(ctx, dataset, VecFieldName, NewSiftHNSWIndex(), false); err != nil {
				panic(err)
			}
		} else if entity.IndexType(indexType) == entity.IvfFlat {
			if err := client.CreateIndex(ctx, dataset, VecFieldName, NewSiftIVFFLATIndex(), false); err != nil {
				panic(err)
			}
		}
	}
	go printCreateIndexProgress(ctx)
	confirmIndexComplete(ctx, client, dataset, VecFieldName)
	return
}

func printCreateIndexProgress(ctx context.Context) {
	ticker := time.NewTicker(time.Second)
	for {
		select {
		case <-ctx.Done():
			fmt.Println("\nCreate index done!")
			return
		case <-ticker.C:
			fmt.Printf("Indexing %s:[%-8d/%-9d]\r", CurIndexType, CurIndexRows, TotalIndexRows)
		}
	}
}

func confirmIndexComplete(ctx context.Context, client milvusClient.Client, dataset string, fieldName string) {
	var err error
	TotalIndexRows, CurIndexRows, err = client.GetIndexBuildProgress(ctx, dataset, fieldName)
	if err != nil {
		panic(err)
	}
	for CurIndexRows != TotalIndexRows {
		TotalIndexRows, CurIndexRows, err = client.GetIndexBuildProgress(ctx, dataset, fieldName)
		if err != nil {
			panic(err)
		}
		time.Sleep(2 * time.Second)
	}
	return
}

func NewTaipHNSWIndex() *entity.IndexHNSW {
	indexParams, err := entity.NewIndexHNSW(entity.L2, 16, 256)
	if err != nil {
		panic(err)
	}
	return indexParams
}

func NewTaipIVFFLATIndex() *entity.IndexIvfFlat {
	indexParams, err := entity.NewIndexIvfFlat(entity.L2, 1024)
	if err != nil {
		panic(err)
	}
	return indexParams
}

func NewSiftHNSWIndex() *entity.IndexHNSW {
	indexParams, err := entity.NewIndexHNSW(entity.L2, 8, 64)
	if err != nil {
		panic(err)
	}
	return indexParams
}

func NewSiftIVFFLATIndex() *entity.IndexIvfFlat {
	indexParams, err := entity.NewIndexIvfFlat(entity.L2, 2048)
	if err != nil {
		panic(err)
	}
	return indexParams
}

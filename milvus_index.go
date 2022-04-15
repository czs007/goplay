package main

import (
	milvusClient "github.com/xiaocai2333/milvus-sdk-go/v2/client"
	"github.com/xiaocai2333/milvus-sdk-go/v2/entity"
)

func CreateIndex(client milvusClient.Client, dataset string, indexType string) {
	if entity.IndexType(indexType) == entity.HNSW {

	}
}

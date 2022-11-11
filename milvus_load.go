package main

import (
	"context"
	"fmt"
	milvusClient "github.com/milvus-io/milvus-sdk-go/v2/client"
	"time"
)

var (
	CurLoadRows = 0
	TotalLoadRows = 0
)

func Load(client milvusClient.Client, dataset string, partitions []string) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go printLoadProgress(ctx)
	if len(partitions) == 0 {
		err := client.LoadCollection(ctx, dataset, false)
		if err != nil {
			print(err)
		}
	}else {
		err := client.LoadPartitions(ctx, dataset, partitions, false)
		if err != nil {
			panic(err)
		}
	}

	confirmLoadComplete(client, dataset)
	return
}

func Release(client milvusClient.Client, dataset string, partitions []string) {
	if len(partitions) == 0 {
		if err := client.ReleaseCollection(context.Background(), dataset); err != nil {
			panic(err)
		}
	}else {
		if err := client.ReleasePartitions(context.Background(), dataset, partitions); err != nil {
			panic(err)
		}
	}
	return
}

func confirmLoadComplete(client milvusClient.Client, dataset string) {
	// TODO: sdk does not implement this function
}

func printLoadProgress(ctx context.Context) {
	ticker := time.NewTicker(time.Second)
	for {
		select {
		case <-ctx.Done():
			fmt.Println("\n Load done!")
			return
		case <-ticker.C:
			fmt.Print("loading...\r")
		}
	}
}

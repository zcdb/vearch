// Copyright 2019 The Vearch Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.

package store

import (
	"context"
	"fmt"
	"strconv"
	"time"

	"github.com/vearch/vearch/v3/internal/config"
	"go.etcd.io/etcd/api/v3/etcdserverpb"
	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/concurrency"
)

func init() {
	Register("etcd", NewEtcdStore)
}

type EtcdStore struct {
	//cli is the etcd client
	cli *clientv3.Client
}

// NewIDGenerate create a global uniqueness id
func (store *EtcdStore) NewIDGenerate(ctx context.Context, key string, base int64, timeout time.Duration) (int64, error) {
	var (
		nextID = int64(0)
		err    error
	)
	err = store.STM(ctx, func(stm concurrency.STM) error {
		v := stm.Get(key)
		if len(v) == 0 {
			stm.Put(key, fmt.Sprintf("%v", base))
			nextID = base
			return nil
		}

		intv, err := strconv.ParseInt(v, 10, 64)
		if err != nil {
			return fmt.Errorf("increment id error in storage :%v", v)
		}

		nextID = intv + 1
		stm.Put(key, strconv.FormatInt(nextID, 10))
		return nil
	})

	if err != nil {
		return int64(0), err
	}
	return nextID, nil
}

func (store *EtcdStore) NewLock(ctx context.Context, key string, timeout time.Duration) *DistLock {
	return NewDistLock(ctx, store.cli, key, timeout)
}

// NewEtcdStore is used to register etcd store init function
func NewEtcdStore(serverAddrs []string) (Store, error) {
	var cli *clientv3.Client
	var err error
	if config.Conf().Global.SupportEtcdAuth {
		cli, err = clientv3.New(clientv3.Config{
			Endpoints:   serverAddrs,
			DialTimeout: 5 * time.Second,
			Username:    config.Conf().EtcdConfig.Username,
			Password:    config.Conf().EtcdConfig.Password,
		})
	} else {
		cli, err = clientv3.New(clientv3.Config{
			Endpoints:   serverAddrs,
			DialTimeout: 5 * time.Second,
		})
	}
	if err != nil {
		return nil, err
	}
	return &EtcdStore{cli: cli}, nil
}

// put kv if already exits it will overwrite
func (store *EtcdStore) Put(ctx context.Context, key string, value []byte) error {
	_, err := store.cli.Put(ctx, key, string(value))
	return err
}

// if key already in , it will check version  if same insert else ?????
// if key is not in , it will put
func (store *EtcdStore) Create(ctx context.Context, key string, value []byte) error {
	resp, err := store.cli.Txn(ctx).
		If(clientv3.Compare(clientv3.Version(key), "=", 0)).
		Then(clientv3.OpPut(key, string(value))).
		Commit()
	if err != nil {
		return err
	}
	if !resp.Succeeded {
		return fmt.Errorf("etcd store key :%v error", key)
	}
	return nil
}

// CreateWithTTL will create the key-value
// if key already in , it will overwrite
// if key is not in , it will put
func (store *EtcdStore) CreateWithTTL(ctx context.Context, key string, value []byte, ttl time.Duration) error {
	if ttl != 0 && int64(ttl.Seconds()) == 0 {
		return fmt.Errorf("ttl time must greater than 1 second")
	}

	grant, err := store.cli.Grant(ctx, int64(ttl.Seconds()))
	if err != nil {
		return err
	}

	_, err = store.cli.Put(ctx, key, string(value), clientv3.WithLease(grant.ID))
	return err
}

func (store *EtcdStore) KeepAlive(ctx context.Context, key string, value []byte, ttl time.Duration) (<-chan *clientv3.LeaseKeepAliveResponse, error) {
	if ttl != 0 && int64(ttl.Seconds()) == 0 {
		return nil, fmt.Errorf("ttl time must greater than 1 second")
	}

	grant, err := store.cli.Grant(ctx, int64(ttl.Seconds()))
	if err != nil {
		return nil, err
	}
	_, err = store.cli.Put(ctx, key, string(value), clientv3.WithLease(grant.ID))
	if err != nil {
		return nil, err
	}

	keepaliveC, err := store.cli.KeepAlive(ctx, grant.ID)
	if err != nil {
		return nil, err
	}

	return keepaliveC, err
}

func (store *EtcdStore) PutWithLeaseId(ctx context.Context, key string, value []byte, ttl time.Duration, leaseId clientv3.LeaseID) error {
	if ttl != 0 && int64(ttl.Seconds()) == 0 {
		return fmt.Errorf("ttl time must greater than 1 second")
	}

	_, err := store.cli.Put(ctx, key, string(value), clientv3.WithLease(leaseId))
	if err != nil {
		return err
	}

	return nil
}

func (store *EtcdStore) Update(ctx context.Context, key string, value []byte) error {
	_, err := store.cli.Put(ctx, key, string(value))
	return err
}

func (store *EtcdStore) Get(ctx context.Context, key string) ([]byte, error) {
	resp, err := store.cli.Get(ctx, key)
	if err != nil {
		return nil, err
	}
	if len(resp.Kvs) < 1 {
		return nil, nil
	}

	return resp.Kvs[0].Value, nil
}

func (store *EtcdStore) PrefixScan(ctx context.Context, prefix string) ([][]byte, [][]byte, error) {
	resp, err := store.cli.Get(ctx, prefix, clientv3.WithPrefix())
	if err != nil {
		return nil, nil, err
	}

	vale := make([][]byte, len(resp.Kvs))
	keys := make([][]byte, len(resp.Kvs))
	for i, v := range resp.Kvs {
		vale[i] = v.Value
		keys[i] = v.Key
	}
	return keys, vale, nil
}

func (store *EtcdStore) Delete(ctx context.Context, key string) error {
	resp, err := store.cli.Delete(ctx, key)
	if err != nil {
		return fmt.Errorf("failed to delete %s from etcd store, the error is :%s", key, err.Error())
	}
	if resp.Deleted != 1 {
		return fmt.Errorf("key not exist error, key:%v", key)
	}
	return nil
}

func (store *EtcdStore) STM(ctx context.Context, apply func(stm concurrency.STM) error) error {
	resp, err := concurrency.NewSTM(store.cli, apply)
	if err != nil {
		return err
	}
	if !resp.Succeeded {
		return fmt.Errorf("etcd stm failed")
	}

	return nil
}

func (store *EtcdStore) WatchPrefix(ctx context.Context, key string) (clientv3.WatchChan, error) {
	startRevision := int64(0)
	initial, err := store.cli.Get(ctx, key)
	if err == nil {
		startRevision = initial.Header.Revision
	}
	watcher := store.cli.Watch(ctx, key, clientv3.WithPrefix(), clientv3.WithRev(startRevision))
	if watcher == nil {
		return nil, fmt.Errorf("watch %v failed", key)
	}

	return watcher, nil
}

func (store *EtcdStore) MemberList(ctx context.Context) (*clientv3.MemberListResponse, error) {
	requestCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	return store.cli.MemberList(requestCtx)
}

func compareEndPoints(eps []string, members []*etcdserverpb.Member) bool {
	memberAddrs := make(map[string]bool)
	for _, member := range members {
		for _, peerURL := range member.PeerURLs {
			memberAddrs[peerURL] = true
		}
	}

	for _, ep := range eps {
		if !memberAddrs[ep] {
			return false
		}
	}

	return true
}

func (store *EtcdStore) MemberStatus(ctx context.Context) ([]*clientv3.StatusResponse, error) {
	membersResp, err := store.cli.MemberList(ctx)
	if err != nil {
		return nil, err
	}
	eps := store.cli.Endpoints()
	if compareEndPoints(eps, membersResp.Members) {
		err := store.cli.Sync(ctx)
		if err != nil {
			return nil, err
		}
	}
	eps = store.cli.Endpoints()
	status := make([]*clientv3.StatusResponse, len(eps))

	for i, ep := range eps {
		requestCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()

		stat, err := store.cli.Status(requestCtx, ep)
		if err != nil {
			status[i] = &clientv3.StatusResponse{
				Header: &etcdserverpb.ResponseHeader{
					ClusterId: 0,
					MemberId:  0,
					Revision:  0,
					RaftTerm:  0,
				},
				Version:          "0",
				DbSizeInUse:      0,
				Leader:           0,
				RaftAppliedIndex: 0,
				Errors:           []string{err.Error()},
			}
		} else {
			status[i] = stat
		}
	}

	return status, nil
}

func (store *EtcdStore) MemberAdd(ctx context.Context, peerAddrs []string) (*clientv3.MemberAddResponse, error) {
	return store.cli.MemberAdd(ctx, peerAddrs)
}

func (store *EtcdStore) MemberRemove(ctx context.Context, id uint64) (*clientv3.MemberRemoveResponse, error) {
	// TODO also remove persistence data
	return store.cli.MemberRemove(ctx, id)
}

func (store *EtcdStore) Endpoints() []string {
	return store.cli.Endpoints()
}

func (store *EtcdStore) MemberSync(ctx context.Context) error {
	return store.cli.Sync(ctx)
}

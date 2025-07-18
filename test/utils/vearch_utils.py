#
# Copyright 2019 The Vearch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

# -*- coding: UTF-8 -*-

import requests
import json
import os
import time
import random
import logging
from multiprocessing import Pool as ThreadPool
import numpy as np
import datetime

router_url = os.getenv("ROUTER_URL", "http://127.0.0.1:9001")
db_name = "ts_db"
space_name = "ts_space"
username = "root"
password = os.getenv("PASSWORD", "secret")

__description__ = """ test utils for vearch """


def process_add_data(items):
    url = router_url + "/document/upsert?timeout=2000000"
    data = {}

    data["documents"] = []
    index = items[0]
    batch_size = items[1]
    features = items[2]
    with_id = items[3]
    full_field = items[4]
    seed = items[5]
    alias_name = items[6]
    partitions = items[7]
    has_string2 = items[8]
    has_vector = items[9]
    db_name = items[10]
    space_name = items[11]

    data["db_name"] = db_name
    data["space_name"] = space_name

    if len(partitions) > 0:
        data["partitions"] = partitions
    if alias_name != "":
        data["space_name"] = alias_name
    for j in range(batch_size):
        param_dict = {}
        if with_id:
            param_dict["_id"] = str(index * batch_size + j)
        param_dict["field_int"] = (index * batch_size + j) * seed
        if has_vector:
            param_dict["field_vector"] = features[j].tolist()
        if full_field:
            param_dict["field_long"] = param_dict["field_int"]
            param_dict["field_float"] = float(param_dict["field_int"])
            param_dict["field_double"] = float(param_dict["field_int"])
            param_dict["field_string"] = str(param_dict["field_int"])
            if has_string2:
                param_dict["field_string2"] = str(param_dict["field_int"] + 1)
        data["documents"].append(param_dict)

    rs = requests.post(url, auth=(username, password), json=data)
    if rs.json()["code"] != 0:
        logger.error(rs.json())

    assert rs.json()["code"] == 0


def add(
    total,
    batch_size,
    xb,
    with_id=False,
    full_field=False,
    seed=1,
    alias_name="",
    partitions=[],
    has_string2=False,
    has_vector=True,
    db_name=db_name,
    space_name=space_name,
):
    pool = ThreadPool()
    total_data = []
    for i in range(total):
        total_data.append(
            (
                i,
                batch_size,
                xb[i * batch_size : (i + 1) * batch_size],
                with_id,
                full_field,
                seed,
                alias_name,
                partitions,
                has_string2,
                has_vector,
                db_name,
                space_name,
            )
        )
    results = pool.map(process_add_data, total_data)
    pool.close()
    pool.join()


def process_add_string_array_data(items):
    url = router_url + "/document/upsert"
    data = {}
    data["db_name"] = db_name
    data["space_name"] = space_name
    data["documents"] = []
    index = items[0]
    batch_size = items[1]
    features = items[2]
    with_id = items[3]
    full_field = items[4]
    seed = items[5]
    alias_name = items[6]
    logger = items[7]
    if items[6] != "":
        data["space_name"] = items[6]
    for j in range(batch_size):
        param_dict = {}
        if with_id:
            param_dict["_id"] = str(index * batch_size + j)
        param_dict["field_int"] = (index * batch_size + j) * seed
        param_dict["field_vector"] = features[j].tolist()
        param_dict["field_string_array"] = [
            str(param_dict["field_int"]),
            str(param_dict["field_int"] + 1000),
        ]
        data["documents"].append(param_dict)

    rs = requests.post(url, auth=(username, password), json=data)
    if logger != None:
        logger.info(rs.json())


def add_string_array(
    total,
    batch_size,
    xb,
    with_id=False,
    full_field=False,
    seed=1,
    alias_name="",
    logger=None,
):
    pool = ThreadPool()
    total_data = []
    for i in range(total):
        total_data.append(
            (
                i,
                batch_size,
                xb[i * batch_size : (i + 1) * batch_size],
                with_id,
                full_field,
                seed,
                alias_name,
                logger,
            )
        )
    results = pool.map(process_add_string_array_data, total_data)
    pool.close()
    pool.join()


def process_add_embedding_size_data(items):
    url = router_url + "/document/upsert?timeout=30000"
    data = {}
    data["documents"] = []

    add_db_name = items[0]
    add_space_name = items[1]
    index = items[2]
    batch_size = items[3]
    embedding_size = items[4]

    data["db_name"] = add_db_name
    data["space_name"] = add_space_name
    for j in range(batch_size):
        param_dict = {}
        param_dict["_id"] = str(index * batch_size + j)
        param_dict["field_int"] = index * batch_size + j
        if embedding_size > 0:
            param_dict["field_vector"] = [
                random.random() for i in range(embedding_size)
            ]
        param_dict["field_long"] = param_dict["field_int"]
        param_dict["field_float"] = float(param_dict["field_int"])
        param_dict["field_double"] = float(param_dict["field_int"])
        param_dict["field_string"] = str(param_dict["field_int"])
        data["documents"].append(param_dict)

    rs = requests.post(url, auth=(username, password), json=data)
    logger.debug(rs.text)
    if embedding_size > 0:
        assert rs.json()["code"] >= 0
    else:
        assert rs.json()["code"] == 0
    return rs


def add_embedding_size(
    db_name, space_name, total, batch_size, embedding_size, max_workers=10
):
    pool = ThreadPool(processes=max_workers)
    total_data = []
    for i in range(total):
        total_data.append((db_name, space_name, i, batch_size, embedding_size))

    results = []
    for i, result in enumerate(pool.imap(process_add_embedding_size_data, total_data)):
        results.append(result)

    pool.close()
    pool.join()
    return results


def process_add_date_data(items):
    url = router_url + "/document/upsert"
    data = {}
    data["documents"] = []

    add_db_name = items[0]
    add_space_name = items[1]
    index = items[2]
    batch_size = items[3]
    embedding_size = items[4]
    date_type = items[5]
    delta = items[6]

    data["db_name"] = add_db_name
    data["space_name"] = add_space_name

    today = datetime.datetime.today().date()
    tomorrow = today + datetime.timedelta(days=1)
    day_after_tomorrow = today + datetime.timedelta(days=2)
    date_str = today.strftime("%Y-%m-%d")
    tomorrow_str = tomorrow.strftime("%Y-%m-%d")
    day_after_tomorrow_str = day_after_tomorrow.strftime("%Y-%m-%d")

    for j in range(batch_size):
        param_dict = {}
        param_dict["_id"] = str(index * batch_size + j)
        param_dict["field_int"] = index * batch_size + j
        param_dict["field_vector"] = [random.random() for i in range(embedding_size)]
        param_dict["field_long"] = param_dict["field_int"]
        param_dict["field_float"] = float(param_dict["field_int"])
        param_dict["field_double"] = float(param_dict["field_int"])
        param_dict["field_string"] = str(param_dict["field_int"])
        if date_type == "str":
            if delta:
                param_dict["field_date"] = tomorrow_str
            else:
                param_dict["field_date"] = date_str
        elif date_type == "timestamp":
            param_dict["field_date"] = int(
                datetime.datetime.strptime(date_str, "%Y-%m-%d")
                .replace(tzinfo=datetime.timezone.utc)
                .timestamp()
            )
        elif date_type == "random":
            param_dict["field_date"] = random.choice(
                [date_str, tomorrow_str, day_after_tomorrow_str]
            )

        data["documents"].append(param_dict)

    response = requests.post(url, auth=(username, password), json=data)
    if response.json()["code"] != 0:
        logger.info(data["documents"][0])
        logger.info(response.json())
    assert response.json()["code"] == 0


def add_date(
    db_name,
    space_name,
    start,
    end,
    batch_size,
    embedding_size,
    date_type,
    delta=0,
):
    pool = ThreadPool()
    total_data = []
    for i in range(start, end):
        total_data.append(
            (
                db_name,
                space_name,
                i,
                batch_size,
                embedding_size,
                date_type,
                delta,
            )
        )
    results = pool.map(process_add_date_data, total_data)
    pool.close()
    pool.join()


def process_add_error_data(items):
    url = router_url + "/document/upsert"
    data = {}
    data["db_name"] = db_name
    data["space_name"] = space_name
    data["documents"] = []
    index = items[0]
    batch_size = items[1]
    features = items[2]
    wrong_number_value = items[3][0]
    wrong_str_value = items[3][1]
    without_vector = items[3][2]
    wrong_db = items[3][3]
    wrong_space = items[3][4]
    wrong_field = items[3][5]
    empty_documents = items[3][6]
    wrong_index_string_length = items[3][7]
    wrong_string_length = items[3][8]
    wrong_vector_type = items[3][9]
    wrong_vector_feature_length = items[3][10]
    wrong_vector_feature_type = items[3][11]
    mismatch_field_type = items[3][12]
    wrong_partition_id = items[3][13]
    max_index_str_length = 1025
    max_str_length = 65536

    if wrong_db:
        data["db_name"] = "wrong_db"
    if wrong_space:
        data["space_name"] = "wrong_space"
    for j in range(batch_size):
        param_dict = {}
        param_dict["field_int"] = index * batch_size + j

        if not without_vector:
            if wrong_vector_type:
                param_dict["field_vector"] = {"feature": features[j].tolist()}
            if wrong_vector_feature_length:
                param_dict["field_vector"] = features[j].tolist()[:1]
            if wrong_vector_feature_type:
                param_dict["field_vector"] = features[j].tolist()[0]
            if (
                not wrong_vector_type
                and not wrong_vector_feature_length
                and not wrong_vector_feature_type
            ):
                param_dict["field_vector"] = features[j].tolist()

        param_dict["field_string"] = str(param_dict["field_int"])
        if wrong_str_value:
            param_dict["field_string"] = float(param_dict["field_int"])

        if wrong_index_string_length:
            param_dict["field_string"] = "".join(
                ["0" for _ in range(max_index_str_length)]
            )

        if wrong_string_length:
            param_dict["field_string1"] = "".join(["0" for _ in range(max_str_length)])

        if wrong_number_value:
            param_dict["field_long"] = param_dict["field_int"]
            param_dict["field_float"] = "field_float"
            param_dict["field_double"] = "field_double"
        else:
            param_dict["field_long"] = param_dict["field_int"]
            param_dict["field_float"] = float(param_dict["field_int"])
            param_dict["field_double"] = float(param_dict["field_int"])
        if wrong_field:
            param_dict["field_wrong"] = param_dict["field_int"]

        if mismatch_field_type:
            param_dict["field_int"] = features[j].tolist()

        if not empty_documents:
            data["documents"].append(param_dict)

        if wrong_partition_id:
            data["partitions"] = [111]

    json_str = json.dumps(data)
    rs = requests.post(url, auth=(username, password), data=json_str)

    if not wrong_string_length:
        logger.info(json_str)
    logger.info(rs.json())

    if "data" in rs.json():
        for result in rs.json()["data"]["document_ids"]:
            assert result["code"] != 0
    else:
        assert rs.status_code != 200


def process_add_mul_error_data(items):
    url = router_url + "/document/upsert"
    data = {}
    data["db_name"] = db_name
    data["space_name"] = space_name
    data["documents"] = []
    index = items[0]
    batch_size = items[1]
    features = items[2]
    parmas_both_wrong, parmas_just_one_wrong, params_just_one_wrong_with_bad_vector = (
        items[3]
    )

    for j in range(batch_size):
        param_dict = {}
        param_dict["field_int"] = index * batch_size + j
        param_dict["field_vector"] = features[j].tolist()

        if parmas_both_wrong:
            param_dict["field_string"] = float(param_dict["field_int"])
        if parmas_just_one_wrong and j == batch_size - 1:
            param_dict["field_string"] = float(param_dict["field_int"])
        if params_just_one_wrong_with_bad_vector and j == batch_size - 1:
            param_dict["field_vector"] = features[j][:-2].tolist()
        data["documents"].append(param_dict)

    json_str = json.dumps(data)
    rs = requests.post(url, auth=(username, password), data=json_str)
    logger.info(rs.json())

    assert rs.status_code != 200


def add_error(total, batch_size, xb, wrong_parameters: list):
    for i in range(total):
        if batch_size == 1:
            process_add_error_data(
                (
                    i,
                    batch_size,
                    xb[i * batch_size : (i + 1) * batch_size],
                    wrong_parameters,
                )
            )
        if batch_size == 2:
            process_add_mul_error_data(
                (
                    i,
                    batch_size,
                    xb[i * batch_size : (i + 1) * batch_size],
                    wrong_parameters,
                )
            )


def process_add_multi_vec_data(items):
    url = router_url + "/document/upsert"
    data = {}
    data["db_name"] = db_name
    data["space_name"] = space_name
    data["documents"] = []
    index = items[0]
    batch_size = items[1]
    features = items[2]
    with_id = items[3]
    full_field = items[4]
    seed = items[5]
    for j in range(batch_size):
        param_dict = {}
        if with_id:
            param_dict["_id"] = str(index * batch_size + j)
        param_dict["field_int"] = (index * batch_size + j) * seed
        param_dict["field_vector"] = features[j].tolist()
        param_dict["field_vector1"] = features[j].tolist()
        if full_field:
            param_dict["field_long"] = param_dict["field_int"]
            param_dict["field_float"] = float(param_dict["field_int"])
            param_dict["field_double"] = float(param_dict["field_int"])
            param_dict["field_string"] = str(param_dict["field_int"])
        data["documents"].append(param_dict)

    json_str = json.dumps(data)
    rs = requests.post(url, auth=(username, password), data=json_str)


def add_multi_vector(total, batch_size, xb, with_id=False, full_field=False, seed=1):
    pool = ThreadPool()
    total_data = []
    for i in range(total):
        total_data.append(
            (
                i,
                batch_size,
                xb[i * batch_size : (i + 1) * batch_size],
                with_id,
                full_field,
                seed,
            )
        )
    results = pool.map(process_add_multi_vec_data, total_data)
    pool.close()
    pool.join()


def process_add_multi_vector_error_data(items):
    url = router_url + "/document/upsert"
    data = {}
    data["db_name"] = db_name
    data["space_name"] = space_name
    data["documents"] = []
    index = items[0]
    batch_size = items[1]
    features = items[2]
    only_one_vector = items[3][0]
    bad_vector_length = items[3][1]

    for j in range(batch_size):
        param_dict = {}
        param_dict["field_int"] = index * batch_size + j

        if only_one_vector:
            param_dict["field_vector"] = features[j].tolist()

        if bad_vector_length:
            param_dict["field_vector"] = features[j].tolist()
            param_dict["field_vector1"] = features[j].tolist()[:1]
        data["documents"].append(param_dict)

    rs = requests.post(url, auth=(username, password), json=data)
    logger.info(rs.json())

    if "data" in rs.json():
        for result in rs.json()["data"]["document_ids"]:
            assert result["code"] != 0
    else:
        assert rs.status_code != 200


def add_multi_vector_error(total, batch_size, xb, wrong_parameters: list):
    for i in range(total):
        process_add_multi_vector_error_data(
            (
                i,
                batch_size,
                xb[i * batch_size : (i + 1) * batch_size],
                wrong_parameters,
            )
        )


def prepare_filter(filter, index, batch_size, seed, full_field):
    if full_field:
        # term_filter = {
        #    "term": {
        #        "field_string": [str(i) for i in range(index * batch_size * seed, (index + 1) * batch_size * seed)]
        #    }
        # }
        # filter.append(term_filter)
        range_filter = [
            {
                "field": "field_int",
                "operator": ">=",
                "value": (index * batch_size) * seed,
            },
            {
                "field": "field_int",
                "operator": "<",
                "value": (index + 1) * batch_size * seed,
            },
            {
                "field": "field_long",
                "operator": ">=",
                "value": (index * batch_size) * seed,
            },
            {
                "field": "field_long",
                "operator": "<",
                "value": (index + 1) * batch_size * seed,
            },
            {
                "field": "field_float",
                "operator": ">=",
                "value": float(index * batch_size * seed),
            },
            {
                "field": "field_float",
                "operator": "<",
                "value": float((index + 1) * batch_size * seed),
            },
            {
                "field": "field_double",
                "operator": ">=",
                "value": float(index * batch_size * seed),
            },
            {
                "field": "field_double",
                "operator": "<",
                "value": float((index + 1) * batch_size * seed),
            },
        ]
        filter.extend(range_filter)
    else:
        range_filter = [
            {
                "field": "field_int",
                "operator": ">=",
                "value": (index * batch_size) * seed,
            },
            {
                "field": "field_int",
                "operator": "<",
                "value": (index + 1) * batch_size * seed,
            },
        ]
        filter.extend(range_filter)


def prepare_wrong_range_filter(filter, index, batch_size):
    range_filter = [
        {
            "field": "field_string",
            "operator": ">",
            "value": [
                str(i) for i in range(index * batch_size, (index + 1) * batch_size)
            ],
        },
    ]
    filter.extend(range_filter)


def prepare_wrong_range_filter_name(filter, index, batch_size):
    range_filter = [
        {"field": "wrong_name", "operator": ">=", "value": index * batch_size},
        {"field": "wrong_name", "operator": "<=", "value": (index + 1) * batch_size},
    ]
    filter.extend(range_filter)


def prepare_wrong_index_filter(filter, index, batch_size):
    range_filter = [
        {"field": "field_long", "operator": ">=", "value": index * batch_size},
        {"field": "field_long", "operator": "<=", "value": (index + 1) * batch_size},
    ]
    filter.extend(range_filter)


def prepare_term_filter(filter, index, batch_size):
    term_filter = [
        {
            "field": "field_string",
            "operator": "IN",
            "value": [
                str(i) for i in range(index * batch_size, (index + 1) * batch_size)
            ],
        },
    ]
    filter.extend(term_filter)


def prepare_wrong_term_filter_name(filter, index, batch_size):
    term_filter = [
        {
            "field": "wrong_name",
            "operator": "IN",
            "value": [
                str(i) for i in range(index * batch_size, (index + 1) * batch_size)
            ],
        },
    ]
    filter.extend(term_filter)


def prepare_wrong_term_filter(filter, index, batch_size):
    term_filter = [
        {"field": "field_int", "operator": "IN", "value": index * batch_size},
        {"field": "field_int", "operator": "IN", "value": (index + 1) * batch_size},
    ]
    filter.extend(term_filter)


def process_query_error_data(items):
    data = {}
    data["db_name"] = db_name
    data["space_name"] = space_name
    index = items[0]
    batch_size = items[1]
    features = items[2]
    logger = items[3]
    interface = items[4]
    url = router_url + "/document/" + interface

    (
        wrong_db,
        wrong_space,
        wrong_id,
        wrong_partition,
        wrong_range_filter,
        wrong_term_filter,
        wrong_filter_index,
        wrong_vector,
        wrong_length_document_ids,
        wrong_both_id_and_filter,
        empty_query,
        empty_document_ids,
        empty_filter,
        wrong_range_filter_name,
        wrong_term_filter_name,
        out_of_bounds_ids,
        wrong_partition_of_bad_type,
        wrong_document_id_of_partition,
        wrong_document_id_of_partition_next,
        wrong_document_id_with_invalid_character,
        wrong_timeout_param,
        timeout,
    ) = items[5]

    max_document_ids_length = 501

    if wrong_db:
        data["db_name"] = "wrong_db"
    if wrong_space:
        data["space_name"] = "wrong_space"

    if wrong_partition and interface == "query":
        data["document_ids"] = ["0"]
        data["partition_id"] = 1008611

    if wrong_id:
        data["document_ids"] = ["wrong_id"]

    if wrong_range_filter:
        data["filters"] = {"operator": "AND", "conditions": []}
        prepare_wrong_range_filter(data["filters"]["conditions"], index, batch_size)
        data["limit"] = batch_size

    if wrong_term_filter:
        data["filters"] = {"operator": "AND", "conditions": []}
        prepare_wrong_term_filter(data["filters"]["conditions"], index, batch_size)
        data["limit"] = batch_size

    if wrong_filter_index:
        data["filters"] = {"operator": "AND", "conditions": []}
        prepare_wrong_index_filter(data["filters"]["conditions"], index, batch_size)
        data["limit"] = batch_size

    if wrong_range_filter_name:
        data["filters"] = {"operator": "AND", "conditions": []}
        prepare_wrong_range_filter_name(
            data["filters"]["conditions"], index, batch_size
        )
        data["limit"] = batch_size

    if wrong_term_filter_name:
        data["filters"] = {"operator": "AND", "conditions": []}
        prepare_wrong_term_filter_name(data["filters"]["conditions"], index, batch_size)
        data["limit"] = batch_size

    if wrong_vector:
        data["vectors"] = []
        vector_info = {
            "field": "field_vector",
            "feature": features[:batch_size].flatten().tolist(),
        }
        data["vectors"].append(vector_info)

    if wrong_length_document_ids:
        data["document_ids"] = [str(i) for i in range(max_document_ids_length)]

    if wrong_both_id_and_filter:
        data["document_ids"] = ["0"]
        data["filters"] = {"operator": "AND", "conditions": []}
        prepare_filter(data["filters"]["conditions"], index, batch_size, 1, True)
        data["limit"] = batch_size

    if empty_document_ids:
        data["document_ids"] = []

    if empty_filter:
        data["filters"] = {"operator": "AND", "conditions": []}

    if out_of_bounds_ids:
        data["document_ids"] = [str(max_document_ids_length + 1)]

    if wrong_partition_of_bad_type and interface == "query":
        data["document_ids"] = ["0"]
        data["partition_id"] = "1008611"

    if wrong_document_id_of_partition and interface == "query":
        data["document_ids"] = ["-1"]
        partition_id = 1
        partition_ids = get_partition(router_url, db_name, space_name)
        partition_id = partition_ids[0]
        data["partition_id"] = partition_id

    if wrong_document_id_of_partition_next and interface == "query":
        data["document_ids"] = ["-2"]
        partition_id = 1
        partition_ids = get_partition(router_url, db_name, space_name)
        partition_id = partition_ids[0]
        data["partition_id"] = partition_id
        data["next"] = True

    if wrong_document_id_with_invalid_character and interface == "query":
        data["document_ids"] = ["aaa"]
        partition_id = 1
        partition_ids = get_partition(router_url, db_name, space_name)
        partition_id = partition_ids[0]
        data["partition_id"] = partition_id

    if wrong_timeout_param:
        url = url + "?timeout=10.5"
        data["document_ids"] = ["0"]

    if timeout and interface == "query":
        url = url + "?timeout=1"
        data["document_ids"] = [str(i) for i in range(100)]

    json_str = json.dumps(data)

    if not wrong_vector:
        logger.info(json_str)

    rs = requests.post(url, auth=(username, password), data=json_str)
    logger.info(rs.json())
    if "data" in rs.json():
        assert rs.json()["data"]["total"] == 0
    else:
        assert rs.status_code != 200


def process_query_multiple_error_data(items):
    data = {}
    data["db_name"] = db_name
    data["space_name"] = space_name
    index = items[0]
    batch_size = items[1]
    features = items[2]
    logger = items[3]
    interface = items[4]
    url = router_url + "/document/" + interface

    (
        params_both_wrong,
        params_just_one_wrong,
        return_both_wrong,
        return_just_one_wrong,
        duplicated_ids,
        duplicated_ids_by_hash,
    ) = items[5]

    if params_both_wrong:
        data["document_ids"] = [1, 2]

    if params_just_one_wrong:
        data["document_ids"] = ["1", 2]

    # not exist so call it return error
    if return_both_wrong:
        data["document_ids"] = ["1008611", "10086"]

    if return_just_one_wrong:
        data["document_ids"] = ["1", "10086"]

    if duplicated_ids:
        data["document_ids"] = ["2", "2"]

    if duplicated_ids_by_hash:
        data["document_ids"] = ["3", "3"]
        data["get_by_hash"] = True

    json_str = json.dumps(data)
    logger.info(json_str)

    rs = requests.post(url, auth=(username, password), data=json_str)
    logger.info(rs.json())

    if params_both_wrong or params_just_one_wrong:
        assert rs.status_code != 200
    else:
        if return_both_wrong:
            assert rs.json()["data"]["total"] == 0
        if return_just_one_wrong:
            assert rs.json()["data"]["total"] == 1
        if duplicated_ids:
            if interface == "query":
                assert rs.json()["data"]["total"] == 2
                duplicated_ids = rs.json()["data"]["documents"]
                assert duplicated_ids[0]["_id"] == "2"
                assert duplicated_ids[1]["_id"] == "2"
            if interface == "delete":
                assert rs.json()["data"]["total"] == 1
                duplicated_ids = rs.json()["data"]["document_ids"]
                assert duplicated_ids[0] == "2"
        if duplicated_ids_by_hash:
            if interface == "query":
                duplicated_ids = rs.json()["data"]["documents"]
                assert rs.json()["data"]["total"] == 2
                assert duplicated_ids[0]["_id"] == "3"
                assert duplicated_ids[1]["_id"] == "3"

            if interface == "delete":
                assert rs.json()["data"]["total"] == 1
                duplicated_ids = rs.json()["data"]["document_ids"]
                assert duplicated_ids[0] == "3"


def query_error(total, batch_size, xb, interface: str, wrong_parameters: list):
    for i in range(total):
        if batch_size == 1:
            process_query_error_data(
                (
                    i,
                    batch_size,
                    xb[i * batch_size : (i + 1) * batch_size],
                    logger,
                    interface,
                    wrong_parameters,
                )
            )
        if batch_size == 2:
            process_query_multiple_error_data(
                (
                    i,
                    batch_size,
                    xb[i * batch_size : (i + 1) * batch_size],
                    logger,
                    interface,
                    wrong_parameters,
                )
            )


def process_query_data(items):
    url = router_url + "/document/query"
    data = {}

    data["vector_value"] = True
    # data["fields"] = ["field_int"]

    logger = items[0]
    index = items[1]
    batch_size = items[2]
    features = items[3]
    full_field = items[4]
    seed = items[5]
    query_type = items[6]
    if items[7] != "":
        data["space_name"] = items[7]

    data["db_name"] = items[8]
    data["space_name"] = items[9]
    check_vector = items[10]
    check = items[11]
    if (
        query_type == "by_partition"
        or query_type == "by_partition_next"
        or query_type == "by_ids"
        or query_type == "by_ids_hash"
    ):
        data["document_ids"] = []
        if query_type == "by_partition_next" and batch_size > 1:
            batch_size -= 1
        for j in range(batch_size):
            data["document_ids"].append(str(index * batch_size + j))

    if query_type == "by_partition" or query_type == "by_partition_next":
        partition_id = 1
        partition_ids = get_partition(router_url, items[8], items[9])
        if len(partition_ids) >= 1:
            partition_id = partition_ids[0]
        logger.debug("partition_id: " + str(partition_id))
        data["partition_id"] = partition_id
        if query_type == "by_partition_next":
            data["next"] = True

    if query_type == "by_filter":
        data["filters"] = {"operator": "AND", "conditions": []}
        prepare_filter(
            data["filters"]["conditions"], index, batch_size, seed, full_field
        )
        data["limit"] = batch_size

    if query_type == "by_ids_hash":
        data["get_by_hash"] = True

    json_str = json.dumps(data)
    rs = requests.post(url, auth=(username, password), data=json_str)

    if not check:
        assert rs.json()["code"] >= 0
        return rs
    if rs.status_code != 200 or "documents" not in rs.json()["data"]:
        logger.info(rs.json())
        logger.info(json_str)

    documents = rs.json()["data"]["documents"]
    if len(documents) != batch_size:
        logger.info("len(documents) = " + str(len(documents)))
        logger.info(rs.json())
        logger.info(json_str)

    assert len(documents) == batch_size
    assert rs.text.find('"total":' + str(batch_size)) >= 0

    for j in range(batch_size):
        value = int(documents[j]["_id"])
        logger.debug(documents[j])
        if (
            query_type == "by_ids"
            or query_type == "by_filter"
            or query_type == "by_ids_hash"
        ):
            assert value == index * batch_size + j
        if query_type == "by_partition_next":
            assert documents[j]["_docid"] == str(index * batch_size + j + 1)

        assert documents[j]["field_int"] == value * seed
        if (query_type == "by_ids" or query_type == "by_ids_hash") and check_vector:
            assert documents[j]["field_vector"] == features[j].tolist()
        if full_field:
            assert documents[j]["field_long"] == value * seed
            assert documents[j]["field_float"] == float(value * seed)
            assert documents[j]["field_double"] == float(value * seed)


def query_interface(
    total,
    batch_size,
    xb,
    full_field=False,
    seed=1,
    query_type="by_ids",
    alias_name="",
    query_db_name=db_name,
    query_space_name=space_name,
    check_vector=True,
    check=True,
):
    if query_type == "by_partition_next" and batch_size == 1:
        total -= 1
    for i in range(total):
        process_query_data(
            (
                logger,
                i,
                batch_size,
                xb[i * batch_size : (i + 1) * batch_size],
                full_field,
                seed,
                query_type,
                alias_name,
                query_db_name,
                query_space_name,
                check_vector,
                check,
            )
        )


# def query_interface(logger, total, batch_size, xb, full_field=False, seed=1, query_type="by_ids"):
#     pool = ThreadPool()
#     total_data = []
#     for i in range(total):
#         total_data.append((logger, i, batch_size,  xb[i * batch_size: (i + 1) * batch_size], full_field, seed, query_type))
#     results = pool.map(process_query_data, total_data)
#     pool.close()
#     pool.join()


def process_delete_data(items):
    url = router_url + "/document/delete?timeout=300000"
    data = {}

    index = items[0]
    batch_size = items[1]
    full_field = items[2]
    seed = items[3]
    delete_type = items[4]

    delete_db_name = items[6]
    delete_space_name = items[7]
    data["db_name"] = delete_db_name
    data["space_name"] = delete_space_name

    if items[5] != "":
        data["space_name"] = items[5]

    check = items[8]

    if delete_type == "by_ids":
        data["document_ids"] = []
        for j in range(batch_size):
            data["document_ids"].append(str(index * batch_size + j))

    if delete_type == "by_filter":
        data["filters"] = {"operator": "AND", "conditions": []}
        prepare_filter(
            data["filters"]["conditions"], index, batch_size, seed, full_field
        )

        data["limit"] = batch_size

    json_str = json.dumps(data)
    rs = requests.post(url, auth=(username, password), data=json_str)
    if not check:
        assert rs.json()["code"] >= 0
        return rs
    if rs.status_code != 200 or "document_ids" not in rs.json()["data"]:
        logger.info(rs.json())
        logger.info(json_str)

    document_ids = rs.json()["data"]["document_ids"]
    if len(document_ids) != batch_size:
        logger.info(
            "batch_size = "
            + str(batch_size)
            + ", len(document_ids) = "
            + str(len(document_ids))
        )
        logger.info(json_str)
        logger.info(rs.json())

    assert len(document_ids) == batch_size
    assert rs.text.find('"total":' + str(batch_size)) >= 0


def delete_interface(
    total,
    batch_size,
    full_field=False,
    seed=1,
    delete_type="by_ids",
    alias_name="",
    delete_db_name=db_name,
    delete_space_name=space_name,
    check=True,
):
    for i in range(total):
        process_delete_data(
            (
                i,
                batch_size,
                full_field,
                seed,
                delete_type,
                alias_name,
                delete_db_name,
                delete_space_name,
                check,
            )
        )


def process_search_data(
    index,
    batch_size,
    features,
    full_field,
    with_filter,
    seed,
    query_type,
    alias_name="",
    db_name="",
    space_name="",
    check=False,
):
    url = router_url + "/document/search?timeout=2000000"
    data = {}
    data["vector_value"] = True

    data["db_name"] = db_name
    data["space_name"] = space_name

    if alias_name != "":
        data["space_name"] = alias_name

    with_symbol = False
    if query_type == "by_vector_with_symbol":
        query_type = "by_vector"
        with_symbol = True

    if query_type == "by_vector":
        data["vectors"] = []
        # logger.debug("partition_id: " + str(partition_id))
        vector_info = {
            "field": "field_vector",
            "feature": features[:batch_size].flatten().tolist(),
        }
        if with_symbol:
            vector_info["symbol"] = "<"
            vector_info["value"] = 40000
        data["vectors"].append(vector_info)

    if with_filter:
        data["filters"] = {"operator": "AND", "conditions": []}
        prepare_filter(
            data["filters"]["conditions"], index, batch_size, seed, full_field
        )

    json_str = json.dumps(data)
    rs = requests.post(url, auth=(username, password), data=json_str)

    if not check:
        assert rs.json()["code"] >= 0
        return rs
    if rs.status_code != 200 or "documents" not in rs.json()["data"]:
        logger.info(rs.json())
        logger.info(json_str)

    documents = rs.json()["data"]["documents"]
    if len(documents) != batch_size:
        logger.info("len(documents) = " + str(len(documents)))
        logger.debug(json_str)
        logger.info(rs.json())

    assert len(documents) == batch_size

    for j in range(batch_size):
        for document in documents[j]:
            value = int(document["_id"])
            logger.debug(value)
            logger.debug(document)
            assert document["field_int"] == value * seed
            if full_field:
                assert document["field_long"] == value * seed
                assert document["field_float"] == float(value * seed)
                assert document["field_double"] == float(value * seed)
            if with_symbol:
                assert document["_score"] < 40000


def search_interface(
    total,
    batch_size,
    xb,
    full_field=False,
    with_filter=False,
    seed=1,
    query_type="by_vector",
    alias_name="",
    search_db_name=db_name,
    search_space_name=space_name,
    check=True,
):
    for i in range(total):
        process_search_data(
            i,
            batch_size,
            xb[i * batch_size : (i + 1) * batch_size],
            full_field,
            with_filter,
            seed,
            query_type,
            alias_name,
            search_db_name,
            search_space_name,
            check,
        )


def create_for_document_test(router_url, embedding_size, properties, partition_num=1):
    space_config = {
        "name": space_name,
        "partition_num": partition_num,
        "replica_num": 1,
        "fields": properties["fields"],
    }
    response = create_db(router_url, db_name)
    logger.info(response.json())
    assert response.json()["code"] == 0

    response = create_space(router_url, db_name, space_config)
    logger.info(response.json())
    assert response.json()["code"] == 0


def prepare_cluster_for_document_test(total, xb, partition_num=1):
    embedding_size = xb.shape[1]
    batch_size = 100
    if total == 0:
        total = xb.shape[0]
    total_batch = int(total / batch_size)
    with_id = True
    full_field = True
    seed = 1

    properties = {}
    properties["fields"] = [
        {
            "name": "field_int",
            "type": "integer",
            "index": {
                "name": "field_int",
                "type": "SCALAR",
            },
        },
        {
            "name": "field_long",
            "type": "long",
        },
        {
            "name": "field_float",
            "type": "float",
        },
        {
            "name": "field_double",
            "type": "double",
            "index": {
                "name": "field_double",
                "type": "SCALAR",
            },
        },
        {
            "name": "field_string",
            "type": "string",
            "index": {
                "name": "field_string",
                "type": "SCALAR",
            },
        },
        {
            "name": "field_vector",
            "type": "vector",
            "index": {
                "name": "gamma",
                "type": "FLAT",
                "params": {
                    "metric_type": "L2",
                },
            },
            "dimension": embedding_size,
            "store_type": "MemoryOnly",
            # "format": "normalization"
        },
    ]

    create_for_document_test(router_url, embedding_size, properties, partition_num)

    add(total_batch, batch_size, xb, with_id, full_field)

    query_interface(total_batch, batch_size, xb, full_field, seed, "by_ids")


def waiting_index_finish(total, timewait=5, space_name=space_name):
    url = router_url + "/dbs/" + db_name + "/spaces/" + space_name
    num = 0
    while num < total:
        num = 0
        response = requests.get(url, auth=(username, password))
        if response.json()["code"] != 0:
            logger.error("waiting index finish, response code is not 0, " + response.text)
            break
        partitions = response.json()["data"]["partitions"]
        for p in partitions:
            num += p["index_num"]
        if num == 0:
            logger.info("waiting index finish, num is 0, "+response.text)
        logger.info("index num: %d" % (num))
        time.sleep(timewait)


def get_space_num():
    url = f"{router_url}/dbs/{db_name}/spaces/{space_name}"
    num = 0
    response = requests.get(url, auth=(username, password))
    num = response.json()["data"]["doc_num"]
    return num


def get_partitions_doc_num(partition_name):
    url = router_url + "/dbs/" + db_name + "/spaces/" + space_name
    num = 0

    response = requests.get(url, auth=(username, password))
    partitions = response.json()["data"]["partitions"]
    for p in partitions:
        if p["name"] == partition_name:
            num += p["doc_num"]
    return num


def search(xq, k: int, batch: bool, query_dict: dict):
    url = router_url + "/document/search?timeout=2000000"

    field_ints = []
    vector_dict = {"vector": [{"field": "field_vector", "feature": []}]}
    if batch:
        vector_dict["vector"][0]["feature"] = xq.flatten().tolist()
        query_dict["vectors"] = vector_dict["vector"]
        json_str = json.dumps(query_dict)
        rs = requests.post(url, auth=(username, password), data=json_str)

        if rs.status_code != 200 or "documents" not in rs.json()["data"]:
            logger.info(rs.json())
            logger.info(json_str)

        for results in rs.json()["data"]["documents"]:
            field_int = []
            for result in results:
                field_int.append(result["field_int"])
            if len(field_int) != k:
                logger.debug("len(field_int)=" + str(len(field_int)))
                [field_int.append(-1) for i in range(k - len(field_int))]
            assert len(field_int) == k
            field_ints.append(field_int)
    else:
        for i in range(xq.shape[0]):
            vector_dict["vector"][0]["feature"] = xq[i].tolist()
            query_dict["vectors"] = vector_dict["vector"]
            json_str = json.dumps(query_dict)
            rs = requests.post(url, auth=(username, password), data=json_str)

            if rs.status_code != 200 or "documents" not in rs.json()["data"]:
                logger.info(rs.json())
                logger.info(json_str)

            field_int = []
            for results in rs.json()["data"]["documents"]:
                for result in results:
                    field_int.append(result["field_int"])
            if len(field_int) != k:
                logger.debug("len(field_int)=" + str(len(field_int)))
                [field_int.append(-1) for i in range(k - len(field_int))]
            assert len(field_int) == k
            field_ints.append(field_int)
    assert len(field_ints) == xq.shape[0]
    return np.array(field_ints)


def evaluate(xq, gt, k, batch, query_dict):
    nq = xq.shape[0]
    t0 = time.time()
    I = search(xq, k, batch, query_dict)
    t1 = time.time()

    recalls = {}
    i = 1
    while i <= k:
        recalls[i] = (I[:, :i] == gt[:, :1]).sum() / float(nq)
        i *= 10

    return (t1 - t0) * 1000.0 / nq, recalls


def drop_db(router_url: str, db_name: str):
    url = f"{router_url}/dbs/{db_name}"
    resp = requests.delete(url, auth=(username, password))
    return resp


def drop_space(router_url: str, db_name: str, space_name: str):
    url = f"{router_url}/dbs/{db_name}/spaces/{space_name}"
    resp = requests.delete(url, auth=(username, password))
    return resp


def destroy(router_url: str, db_name: str, space_name: str):
    drop_space(router_url, db_name, space_name)
    drop_db(router_url, db_name)


def create_space(router_url: str, db_name: str, space_config: dict):
    url = f"{router_url}/dbs/{db_name}/spaces?timeout=1000000"
    resp = requests.post(url, auth=(username, password), json=space_config)
    return resp


def update_space_partition(
    router_url: str, db_name: str, space_name: str, partition_num: int
):
    url = f"{router_url}/dbs/{db_name}/spaces/{space_name}"
    data = {"partition_num": partition_num}
    resp = requests.put(url, auth=(username, password), json=data)
    return resp


def update_space_partition_rule(
    router_url: str,
    db_name: str,
    space_name: str,
    partition_name: str = None,
    operator_type: str = None,
    partition_rule: dict = None,
):
    url = f"{router_url}/dbs/{db_name}/spaces/{space_name}"
    data = {}
    if partition_name is not None:
        data["partition_name"] = partition_name
    if operator_type is not None:
        data["operator_type"] = operator_type
    if partition_rule is not None:
        data["partition_rule"] = partition_rule["partition_rule"]
    resp = requests.put(url, auth=(username, password), json=data)
    return resp


def get_space(router_url: str, db_name: str, space_name: str, detail: bool = False):
    if detail:
        url = f"{router_url}/dbs/{db_name}/spaces/{space_name}?detail=true"
    else:
        url = f"{router_url}/dbs/{db_name}/spaces/{space_name}"
    resp = requests.get(url, auth=(username, password))
    return resp


def get_partition(router_url: str, db_name: str, space_name: str):
    url = f"{router_url}/cache/dbs/{db_name}/spaces/{space_name}"
    resp = requests.get(url, auth=(username, password))
    partition_infos = resp.json()["data"]["partitions"]
    partition_ids = []
    for partition_info in partition_infos:
        partition_ids.append(partition_info["id"])
    return partition_ids


def get_space_cache(router_url: str, db_name: str, space_name: str):
    url = f"{router_url}/cache/dbs/{db_name}/spaces/{space_name}"
    resp = requests.get(url, auth=(username, password))
    return resp


def get_router_info(router_url: str):
    url = f"{router_url}"
    resp = requests.get(url, auth=(username, password))
    return resp


def get_cluster_stats(router_url: str):
    url = f"{router_url}/cluster/stats"
    resp = requests.get(url, auth=(username, password))
    return resp


def get_cluster_health(
    router_url: str, db_name: str = "", space_name: str = "", detail: bool = True
):
    url = f"{router_url}/cluster/health"
    if detail:
        url = url + "?detail=true"
    else:
        url = url + "?detail=false"
    if db_name != "":
        url = url + "&db=" + db_name
    if space_name != "":
        url = url + "&space=" + space_name
    resp = requests.get(url, auth=(username, password))
    return resp


def get_cluster_member_list(router_url: str):
    url = f"{router_url}/members"
    resp = requests.get(url, auth=(username, password))
    return resp


def get_cluster_member_stats(router_url: str):
    url = f"{router_url}/members/stats"
    resp = requests.get(url, auth=(username, password))
    return resp


def remove_cluster_member(router_url: str, member_id: int, master_name: str):
    url = f"{router_url}/members"
    data = {"id": member_id, "name": master_name}
    resp = requests.delete(url, auth=(username, password), json=data)
    return resp


def add_cluster_member(router_url: str, peer_addrs: list):
    url = f"{router_url}/members"
    data = {"peer_addrs": peer_addrs}
    resp = requests.post(url, auth=(username, password), json=data)
    return resp


def get_servers_status(router_url: str):
    url = f"{router_url}/servers"
    resp = requests.get(url, auth=(username, password))
    return resp


def get_cluster_partition(router_url: str):
    url = f"{router_url}/partitions"
    resp = requests.get(url, auth=(username, password))
    return resp


def change_partitons(router_url: str, pids: list, node_id: int, method: int):
    url = f"{router_url}/partitions/change_member"
    data = {"partition_ids": pids, "node_id": node_id, "method": method}
    resp = requests.post(url, auth=(username, password), json=data)
    return resp


def get_cluster_version(router_url: str):
    url = f"{router_url}/"
    resp = requests.get(url, auth=(username, password))
    return resp


def list_dbs(router_url: str):
    url = f"{router_url}/dbs"
    resp = requests.get(url, auth=(username, password))
    return resp


def create_db(router_url: str, db_name: str):
    url = f"{router_url}/dbs/" + db_name
    resp = requests.post(url, auth=(username, password))
    return resp


def get_db(router_url: str, db_name: str):
    url = f"{router_url}/dbs/{db_name}"
    resp = requests.get(url, auth=(username, password))
    return resp


def list_spaces(router_url: str, db_name: str):
    url = f"{router_url}/dbs/{db_name}/spaces?detail=true"
    resp = requests.get(url, auth=(username, password))
    return resp


def describe_space(router_url: str, db_name: str, space_name: str):
    url = f"{router_url}/dbs/{db_name}/spaces/{space_name}?detail=true"
    try:
        resp = requests.get(url, auth=(username, password))
        return resp
    except Exception as e:
        logger.error(e)


def index_rebuild(
    router_url: str,
    db_name: str,
    space_name: str,
    drop_before_rebuild: bool = True,
    partition_id: int = 0,
):
    url = f"{router_url}/index/rebuild"
    data = {
        "db_name": db_name,
        "space_name": space_name,
        "drop_before_rebuild": drop_before_rebuild,
        "partition_id": partition_id,
    }
    logger.info(data)
    resp = requests.post(url, auth=(username, password), json=data)
    return resp


def create_alias(router_url: str, alias_name: str, db_name: str, space_name: str):
    url = f"{router_url}/alias/{alias_name}/dbs/{db_name}/spaces/{space_name}"
    resp = requests.post(url, auth=(username, password))
    return resp


def update_alias(router_url: str, alias_name: str, db_name: str, space_name: str):
    url = f"{router_url}/alias/{alias_name}/dbs/{db_name}/spaces/{space_name}"
    resp = requests.put(url, auth=(username, password))
    return resp


def get_alias(router_url: str, alias_name: str):
    url = f"{router_url}/alias/{alias_name}"
    resp = requests.get(url, auth=(username, password))
    return resp


def get_all_alias(router_url: str):
    url = f"{router_url}/alias"
    resp = requests.get(url, auth=(username, password))
    return resp


def drop_alias(router_url: str, alias_name: str):
    url = f"{router_url}/alias/{alias_name}"
    resp = requests.delete(url, auth=(username, password))
    return resp


def create_user(
    router_url: str, user_name: str, user_password: str = None, role_name: str = None
):
    url = f"{router_url}/users"
    data = {"name": user_name}
    if role_name is not None:
        data["role_name"] = role_name
    if user_password is not None:
        data["password"] = user_password
    resp = requests.post(url, json=data, auth=(username, password))
    return resp


def set_password(value):
    global password
    password = value


def update_user(
    router_url: str,
    user_name: str,
    new_password: str = None,
    old_password: str = None,
    role_name: str = None,
    auth_user: str = None,
    auth_password: str = None,
):
    url = f"{router_url}/users"
    data = {"name": user_name}
    if new_password is not None:
        data["password"] = new_password
    if old_password is not None:
        data["old_password"] = old_password
    if role_name is not None:
        data["role_name"] = role_name
    if auth_user is not None and auth_password is not None:
        resp = requests.put(url, json=data, auth=(auth_user, auth_password))
        return resp
    else:
        resp = requests.put(url, json=data, auth=(username, password))
        return resp


def get_user(router_url: str, user_name: str):
    url = f"{router_url}/users/{user_name}"
    resp = requests.get(url, auth=(username, password))
    return resp


def get_all_users(router_url: str):
    url = f"{router_url}/users"
    resp = requests.get(url, auth=(username, password))
    return resp


def drop_user(router_url: str, user_name: str):
    url = f"{router_url}/users/{user_name}"
    resp = requests.delete(url, auth=(username, password))
    return resp


def create_role(router_url: str, role_name: str, privileges: dict):
    url = f"{router_url}/roles"
    data = {"name": role_name, "privileges": privileges}
    resp = requests.post(url, json=data, auth=(username, password))
    return resp


def change_role_privilege(
    router_url: str, role_name: str, operator: str, privileges: dict
):
    url = f"{router_url}/roles"
    data = {"name": role_name, "operator": operator, "privileges": privileges}
    resp = requests.put(url, json=data, auth=(username, password))
    return resp


def get_role(router_url: str, role_name: str):
    url = f"{router_url}/roles/{role_name}"
    resp = requests.get(url, auth=(username, password))
    return resp


def get_cache_role(router_url: str, role_name: str):
    url = f"{router_url}/cache/roles/{role_name}"
    resp = requests.get(url, auth=(username, password))
    return resp


def get_all_roles(router_url: str):
    url = f"{router_url}/roles"
    resp = requests.get(url, auth=(username, password))
    return resp


def drop_role(router_url: str, role_name: str):
    url = f"{router_url}/roles/{role_name}"
    resp = requests.delete(url, auth=(username, password))
    return resp


def server_resource_limit(
    router_url: str,
    resource_exhausted: bool = None,
    rate: float = None,
    logger=None,
):
    url = f"{router_url}/partitions/resource_limit"
    data = {}
    if resource_exhausted is not None:
        data["resource_exhausted"] = resource_exhausted
    if rate is not None:
        data["rate"] = rate
    if logger is not None:
        logger.info(data)
    resp = requests.post(url, auth=(username, password), json=data)
    return resp


def init_log():
    log = logging.getLogger("test")
    log.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s"
    )
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)
    return log


logger = init_log()

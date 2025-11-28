import great_expectations as gx
import great_expectations.expectations as gxe

context = gx.get_context(mode="file")

if not context.is_project_initialized("gx"):
    data_path = "./prepared_data/"
    data_source_name = "local_prepared_data"
    data_source = context.data_sources.add_pandas_filesystem(
        name=data_source_name, base_directory=data_path,
    )

    file_names = ["train.csv", "val.csv", "test.csv"]

    for file_name in file_names:
        asset_name = file_name.replace(".", "_") + "_asset"
        asset = data_source.add_csv_asset(name=asset_name)
        asset.add_batch_definition_path(name=file_name, path=file_name)


    expectations = []
    for column in asset.batch_definitions[0].get_batch().columns():
        expectations.append(
            gxe.ExpectColumnValuesToNotBeNull(column=column)
        )
        if column.startswith("ohe__"):
            expectations.append(
                gxe.ExpectColumnDistinctValuesToBeInSet(column=column, value_set={0, 1})
            )
        if column.startswith("numeric_preprocessor__"):
            expectations.extend([
                gxe.ExpectColumnValuesToBeBetween(
                    column=column, min_value=0, max_value=1
                ),
            ])

    suite_name = "suite"
    suite = gx.ExpectationSuite(
        name=suite_name,
        expectations=expectations,
    )
    suite = context.suites.add(suite)

    for asset in data_source.get_assets_as_dict().values():
        batch_definition = asset.batch_definitions[0]
        val_definition = gx.ValidationDefinition(
            name=batch_definition.name + "_val_def", data=batch_definition, suite=suite
        )
        context.validation_definitions.add(val_definition)
    
    site_name = "site"
    context.add_data_docs_site(site_name=site_name, site_config={
        "class_name": "SiteBuilder",
        "site_index_builder": {"class_name": "DefaultSiteIndexBuilder"},
        "store_backend": {
            "class_name": "TupleFilesystemStoreBackend",
            "base_directory": "uncommitted/data_docs/local_site/",
        },
    })
    
    checkpoint = context.checkpoints.add(
        gx.Checkpoint(
            name="all_data_validation",
            validation_definitions=context.validation_definitions.all(),
            actions=[
                gx.checkpoint.actions.UpdateDataDocsAction(
                    name="update_my_site", site_names=[site_name]
                )
            ]
        )
    )

    results = checkpoint.run()
    
else:
    checkpoint = context.checkpoints.get("all_data_validation")
    results = checkpoint.run()

if not dict(results)["success"]:
    raise Exception("Validation Failed.")
print("Validation Succeeded.")

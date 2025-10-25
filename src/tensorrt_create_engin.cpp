//未开始工作


{
    IBuilder             *builder = createInferBuilder(gLogger);
    INetworkDefinition   *network = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    IOptimizationProfile *profile = builder->createOptimizationProfile();
    IBuilderConfig       *config  = builder->createBuilderConfig();
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 30);

    ITensor *inputTensor = network->addInput("inputT0", DataType::kFLOAT, Dims32 {3, {-1, -1, -1}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims32 {3, {1, 1, 1}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims32 {3, {3, 4, 5}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims32 {3, {6, 8, 10}});
    config->addOptimizationProfile(profile);

    IIdentityLayer *identityLayer = network->addIdentity(*inputTensor);
    network->markOutput(*identityLayer->getOutput(0));
    IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
    if (engineString == nullptr || engineString->size() == 0)
    {
        std::cout << "Failed building serialized engine!" << std::endl;
        return;
    }
    std::cout << "Succeeded building serialized engine!" << std::endl;

    IRuntime *runtime {createInferRuntime(gLogger)};
    engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
    if (engine == nullptr)
    {
        std::cout << "Failed building engine!" << std::endl;
        return;
    }
    std::cout << "Succeeded building engine!" << std::endl;

    std::ofstream engineFile(trtFile, std::ios::binary);
    if (!engineFile)
    {
        std::cout << "Failed opening file to write" << std::endl;
        return;
    }
    engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
    if (engineFile.fail())
    {
        std::cout << "Failed saving .plan file!" << std::endl;
        return;
    }
    std::cout << "Succeeded saving .plan file!" << std::endl;
}

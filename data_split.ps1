# 读取 test 列表
$testList = Get-Content "D:\casia_files\hcp\hcp1200\split_test.txt" |
    Where-Object { $_ -match '^\s*\d{6}/?\s*$' } |
    ForEach-Object { ($_ -replace '[^0-9]').Trim() }

# 定义路径
$baseDir = "F:\HCP1200_split"
$testDir = Join-Path $baseDir "test"
$trainValidDir = Join-Path $baseDir "train_valid"

# 创建 test 和 train_valid 文件夹（如果不存在）
New-Item -ItemType Directory -Path $testDir -Force | Out-Null
New-Item -ItemType Directory -Path $trainValidDir -Force | Out-Null

# 获取所有一级子目录名
$allSubDirs = Get-ChildItem -Path $baseDir -Directory | Where-Object {
    $_.Name -ne "test" -and $_.Name -ne "train_valid"
}

# 分类移动
foreach ($dir in $allSubDirs) {
    if ($testList -contains $dir.Name) {
        Move-Item -Path $dir.FullName -Destination $testDir
        Write-Host "Moved $($dir.Name) to test"
    } else {
        Move-Item -Path $dir.FullName -Destination $trainValidDir
        Write-Host "Moved $($dir.Name) to train_valid"
    }
}


# 定义路径
$splitFile = "D:\casia_files\hcp\hcp1200\split_train_valid.txt"
$trainValidBase = "F:\HCP1200_split\train_valid"

# 初始化变量
$fold = ""
$mapping = @{}

# 逐行解析 split 文件
Get-Content $splitFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -match "^fold_\d+/?$") {
        $fold = $line -replace "/", ""
        $mapping[$fold] = @()
    } elseif ($line -match "^\d{6}/?$") {
        $subject = $line -replace "/", ""
        if ($fold -ne "") {
            $mapping[$fold] += $subject
        }
    }
}

# 遍历每个 fold，创建文件夹并移动对应数据
foreach ($foldName in $mapping.Keys) {
    $foldPath = Join-Path $trainValidBase $foldName
    New-Item -ItemType Directory -Path $foldPath -Force | Out-Null

    foreach ($subject in $mapping[$foldName]) {
        $sourcePath = Join-Path $trainValidBase $subject
        $destPath = Join-Path $foldPath $subject
        if (Test-Path $sourcePath) {
            Move-Item -Path $sourcePath -Destination $foldPath
            Write-Host "Moved $subject to $foldName"
        } else {
            Write-Warning "$subject not found in train_valid"
        }
    }
}

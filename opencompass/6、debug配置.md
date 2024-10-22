# 1、debug配置
launch.json
```
{
    // 使用 IntelliSense 了解相关属性。
    // 悬停以查看现有属性的描述。
    // 欲了解更多信息，请访问: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",

            "cwd": "${workspaceFolder}",
            "env": {"PYTHONPATH": "${workspaceRoot}"},
            "console": "internalConsole",
            // "console": "integratedTerminal",
            "python": "${command:python.interpreterPath}",
            "args": [
                "${workspaceFolder}/configs/eval_modelscope_chatglmv1.py",
                "-r",
                "${workspaceFolder}/outputs/20241018" 
                ],
                
            "justMyCode": false,
        }
    ]
}
```

# 2、debug记录

# 对比图 — 「奶油肌」批处理的完整对比

两批样张经 `retouch batch --preset cream`（0.6.3）处理后的逐张对比，由 `python tools/sample_montages.py compare_all` 生成。

| 目录 | 内容 |
|---|---|
| `原片/` | 7 张原片：原图 \| 本程序 \| 像素蛋糕「奶油肌」预设导出（doc/test_report_batch.md） |
| `批量测试/` | 15 张批量测试样张：原图 \| 本程序；1V3A2954 有像素蛋糕的导出，多一列（doc/test_report_batch_test.md） |

每张照片一张 `<照片>_全图.jpg`（全图，每块长边 2400 像素）；每张脸一张 `<照片>_脸<k>.jpg`（人脸框的 1.5 倍）
与一张 `<照片>_脖子<k>.jpg`（人脸框下方），每块宽 800 像素，k 按人脸检测的顺序从 1 编号。人脸检测得分低于 0.95 的
误检（1V3A2922 的花束、1V3A2886 与 1V3A2889 裙摆上的亮片）不出局部图。身体皮肤阴影色块的修正见
doc/test_report_shadow_blotches.md。

重新生成（原片与像素蛋糕导出放在 `test/`，不入库）：

```sh
retouch batch -i test/原片 -o out/batch --preset cream --landmarks-dir out/batch/_landmarks --overwrite
retouch batch -i test/批量测试 -o out/batch_test --preset cream --landmarks-dir out/batch_test/_landmarks --overwrite
python tools/sample_montages.py compare_all
```

from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class HeadHook(Hook):

    def __init__(self, interval=50):
        self.interval = interval

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        if self.every_n_iters(runner, self.interval):
            model = runner.model
            runner.logger.info(
                'Differing_Elements_Even_Count: '
                f'{(model.module.bbox_head.rfassigner.differing_elements_count)} '
                )
            model.module.bbox_head.rfassigner.differing_elements_count = 0
            # runner.logger.info(
            #     'RF0_Scale: '
            #     f'{(model.module.bbox_head.rfassigner.ratio[0].item())} '
            #     )
            # runner.logger.info(
            #     'RF1_Scale: '
            #     f'{(model.module.bbox_head.rfassigner.ratio[1].item())} '
            #     )
            # runner.logger.info(
            #     'RF2_Scale: '
            #     f'{(model.module.bbox_head.rfassigner.ratio[2].item())} '
            #     )
            # runner.logger.info(
            #     'RF3_Scale: '
            #     f'{(model.module.bbox_head.rfassigner.ratio[3].item())} '
            #     )
        # pass

    def after_iter(self, runner):
        pass